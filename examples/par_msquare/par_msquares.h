// MSQUARES :: https://github.com/prideout/par
// Converts fp32 grayscale images, or 8-bit color images, into triangles.
//
// For grayscale images, a threshold is specified to determine insideness.
// For color images, an exact color is specified to determine insideness.
// Color images can be r8, rg16, rgb24, or rgba32. For a visual overview of
// the API and all the flags, see:
//
//     http://github.prideout.net/marching-cubes/
//
// The MIT License
// Copyright (c) 2015 Philip Rideout

#include <stdint.h>

// -----------------------------------------------------------------------------
// BEGIN PUBLIC API
// -----------------------------------------------------------------------------

typedef uint8_t par_byte;

typedef struct par_msquares_meshlist_s par_msquares_meshlist;

// Encapsulates the results of a marching squares operation.
typedef struct {
    float* points;        // pointer to XY (or XYZ) vertex coordinates
    int npoints;          // number of vertex coordinates
    uint16_t* triangles;  // pointer to 3-tuples of vertex indices
    int ntriangles;       // number of 3-tuples
    int dim;              // number of floats per point (either 2 or 3)
    int nconntriangles;   // internal use only
} par_msquares_mesh;

// Reverses the "insideness" test.
#define PAR_MSQUARES_INVERT (1 << 0)

// Returns a meshlist with two meshes: one for the inside, one for the outside.
#define PAR_MSQUARES_DUAL (1 << 1)

// Returned meshes have 3-tuple coordinates instead of 2-tuples. When using
// from_color, the Z coordinate represents the alpha value of the color.  With
// from_grayscale, the Z coordinate represents the value of the nearest pixel in
// the source image.
#define PAR_MSQUARES_HEIGHTS (1 << 2)

// Applies a step function to the Z coordinates.  Requires HEIGHTS and DUAL.
#define PAR_MSQUARES_SNAP (1 << 3)

// Adds extrusion triangles to each mesh other than the lowest mesh.  Requires
// the PAR_MSQUARES_HEIGHTS flag to be present.
#define PAR_MSQUARES_CONNECT (1 << 4)

// Enables quick & dirty (not best) simpification of the returned mesh.
#define PAR_MSQUARES_SIMPLIFY (1 << 5)

par_msquares_meshlist* par_msquares_from_grayscale(float const* data, int width,
    int height, int cellsize, float threshold, int flags);

par_msquares_meshlist* par_msquares_from_color(par_byte const* data, int width,
    int height, int cellsize, uint32_t color, int bpp, int flags);

typedef int (*par_msquares_inside_fn)(int, void*);
typedef float (*par_msquares_height_fn)(float, float, void*);

par_msquares_meshlist* par_msquares_from_function(int width, int height,
    int cellsize, int flags, void* context, par_msquares_inside_fn insidefn,
    par_msquares_height_fn heightfn);

par_msquares_mesh const* par_msquares_get_mesh(par_msquares_meshlist*, int n);

int par_msquares_get_count(par_msquares_meshlist*);

void par_msquares_free(par_msquares_meshlist*);

// -----------------------------------------------------------------------------
// END PUBLIC API
// -----------------------------------------------------------------------------

#ifdef PAR_MSQUARES_IMPLEMENTATION

#include <stdlib.h>
#include <assert.h>
#include <float.h>

#define MIN(a, b) (a > b ? b : a)
#define MAX(a, b) (a > b ? a : b)
#define CLAMP(v, lo, hi) MAX(lo, MIN(hi, v))

struct par_msquares_meshlist_s {
    int nmeshes;
    par_msquares_mesh** meshes;
};

static int** point_table = 0;
static int** triangle_table = 0;

static void par_init_tables()
{
    point_table = (int**)malloc(16 * sizeof(int*));
    triangle_table = (int**)malloc(16 * sizeof(int*));

    char const* CODE_TABLE =
        "0 0\n"
        "1 1 0 1 7\n"
        "2 1 1 2 3\n"
        "3 2 0 2 3 3 7 0\n"
        "4 1 7 5 6\n"
        "5 2 0 1 5 5 6 0\n"
        "6 2 1 2 3 7 5 6\n"
        "7 3 0 2 3 0 3 5 0 5 6\n"
        "8 1 3 4 5\n"
        "9 4 0 1 3 0 3 4 0 4 5 0 5 7\n"
        "a 2 1 2 4 4 5 1\n"
        "b 3 0 2 4 0 4 5 0 5 7\n"
        "c 2 7 3 4 4 6 7\n"
        "d 3 0 1 3 0 3 4 0 4 6\n"
        "e 3 1 2 4 1 4 6 1 6 7\n"
        "f 2 0 2 4 4 6 0\n";

    char const* table_token = CODE_TABLE;
    for (int i = 0; i < 16; i++) {
        char code = *table_token;
        assert(i == code - (code >= 'a' ? ('a' - 0xa) : '0'));
        table_token += 2;
        int ntris = *table_token - '0';
        table_token += 2;
        int* sqrtris = triangle_table[i] =
                (int*)malloc((ntris + 1) * 3 * sizeof(int));
        sqrtris[0] = ntris;
        int mask = 0;
        int* sqrpts = point_table[i] = (int*)malloc(7 * sizeof(int));
        sqrpts[0] = 0;
        for (int j = 0; j < ntris * 3; j++, table_token += 2) {
            int midp = *table_token - '0';
            int bit = 1 << midp;
            if (!(mask & bit)) {
                mask |= bit;
                sqrpts[++sqrpts[0]] = midp;
            }
            sqrtris[j + 1] = midp;
        }
    }
}

typedef struct {
    float const* data;
    float threshold;
    int width;
    int height;
} gray_context;

static int gray_inside(int location, void* contextptr)
{
    gray_context* context = (gray_context*) contextptr;
    return context->data[location] > context->threshold;
}

static float gray_height(float x, float y, void* contextptr)
{
    gray_context* context = (gray_context*) contextptr;
    int i = CLAMP(context->width * x, 0, context->width - 1);
    int j = CLAMP(context->height * y, 0, context->height - 1);
    return context->data[i + j * context->width];
}

typedef struct {
    par_byte const* data;
    par_byte color[4];
    int bpp;
    int width;
    int height;
} color_context;

static int color_inside(int location, void* contextptr)
{
    color_context* context = (color_context*) contextptr;
    par_byte const* data = context->data + location * context->bpp;
    for (int i = 0; i < context->bpp; i++) {
        if (data[i] != context->color[i]) {
            return 0;
        }
    }
    return 1;
}

static float color_height(float x, float y, void* contextptr)
{
    color_context* context = (color_context*) contextptr;
    assert(context->bpp == 4);
    int i = CLAMP(context->width * x, 0, context->width - 1);
    int j = CLAMP(context->height * y, 0, context->height - 1);
    int k = i + j * context->width;
    return context->data[k * 4 + 3] / 255.0;
}

par_msquares_meshlist* par_msquares_from_color(par_byte const* data, int width,
    int height, int cellsize, uint32_t color, int bpp, int flags)
{
    color_context context;
    context.bpp = bpp;
    context.color[0] = (color >> 16) & 0xff;
    context.color[1] = (color >> 8) & 0xff;
    context.color[2] = (color & 0xff);
    context.color[3] = (color >> 24) & 0xff;
    context.data = data;
    context.width = width;
    context.height = height;
    return par_msquares_from_function(
        width, height, cellsize, flags, &context, color_inside, color_height);
}

par_msquares_meshlist* par_msquares_from_grayscale(float const* data, int width,
    int height, int cellsize, float threshold, int flags)
{
    gray_context context;
    context.width = width;
    context.height = height;
    context.data = data;
    context.threshold = threshold;
    return par_msquares_from_function(
        width, height, cellsize, flags, &context, gray_inside, gray_height);
}

par_msquares_mesh const* par_msquares_get_mesh(
    par_msquares_meshlist* mlist, int mindex)
{
    assert(mlist && mindex < mlist->nmeshes);
    return mlist->meshes[mindex];
}

int par_msquares_get_count(par_msquares_meshlist* mlist)
{
    assert(mlist);
    return mlist->nmeshes;
}

void par_msquares_free(par_msquares_meshlist* mlist)
{
    par_msquares_mesh** meshes = mlist->meshes;
    for (int i = 0; i < mlist->nmeshes; i++) {
        free(meshes[i]->points);
        free(meshes[i]->triangles);
        free(meshes[i]);
    }
    free(meshes);
    free(mlist);
}

// Combine multiple meshlists by moving mesh pointers, and optionally apply
// a "snap" operation that assigns a single Z value across all verts in each
// mesh.  The Z value determined by the mesh's position in the final mesh list.
static par_msquares_meshlist* par_msquares_merge(par_msquares_meshlist** lists,
    int count, int snap)
{
    par_msquares_meshlist* merged = (par_msquares_meshlist*)malloc(sizeof(par_msquares_meshlist));
    merged->nmeshes = 0;
    for (int i = 0; i < count; i++) {
        merged->nmeshes += lists[i]->nmeshes;
    }
    merged->meshes = (par_msquares_mesh**)malloc(sizeof(par_msquares_mesh*) * merged->nmeshes);
    par_msquares_mesh** pmesh = merged->meshes;
    for (int i = 0; i < count; i++) {
        par_msquares_meshlist* meshlist = lists[i];
        for (int j = 0; j < meshlist->nmeshes; j++) {
            *pmesh++ = meshlist->meshes[j];
        }
        free(meshlist);
    }
    if (!snap) {
        return merged;
    }
    pmesh = merged->meshes;
    float zmin = FLT_MAX;
    float zmax = -zmin;
    for (int i = 0; i < merged->nmeshes; i++, pmesh++) {
        float* pzed = (*pmesh)->points + 2;
        for (int j = 0; j < (*pmesh)->npoints; j++, pzed += 3) {
            zmin = MIN(*pzed, zmin);
            zmax = MAX(*pzed, zmax);
        }
    }
    float zextent = zmax - zmin;
    pmesh = merged->meshes;
    for (int i = 0; i < merged->nmeshes; i++, pmesh++) {
        float* pzed = (*pmesh)->points + 2;
        float zed = zmin + zextent * i / (merged->nmeshes - 1);
        for (int j = 0; j < (*pmesh)->npoints; j++, pzed += 3) {
            *pzed = zed;
        }
    }
    if (!(snap & PAR_MSQUARES_CONNECT)) {
        return merged;
    }
    for (int i = 1; i < merged->nmeshes; i++) {
        par_msquares_mesh* mesh = merged->meshes[i];

        // Find all extrusion points.  This is tightly coupled to the
        // tessellation code, which generates two "connector" triangles for each
        // extruded edge.  The first two verts of the second triangle are the
        // verts that need to be displaced.
        char* markers = (char*)calloc(mesh->npoints, 1);
        int tri = mesh->ntriangles - mesh->nconntriangles;
        while (tri < mesh->ntriangles) {
            markers[mesh->triangles[tri * 3 + 3]] = 1;
            markers[mesh->triangles[tri * 3 + 4]] = 1;
            tri += 2;
        }

        // Displace all extrusion points down to the previous level.
        float zed = zmin + zextent * (i - 1) / (merged->nmeshes - 1);
        float* pzed = mesh->points + 2;
        for (int j = 0; j < mesh->npoints; j++, pzed += 3) {
            if (markers[j]) {
                *pzed = zed;
            }
        }
        free(markers);
    }
    return merged;
}

par_msquares_meshlist* par_msquares_from_function(int width, int height,
    int cellsize, int flags, void* context, par_msquares_inside_fn insidefn,
    par_msquares_height_fn heightfn)
{
    assert(width > 0 && width % cellsize == 0);
    assert(height > 0 && height % cellsize == 0);

    if (flags & PAR_MSQUARES_DUAL) {
        int connect = flags & PAR_MSQUARES_CONNECT;
        int snap = flags & PAR_MSQUARES_SNAP;
        int heights = flags & PAR_MSQUARES_HEIGHTS;
        if (!heights) {
            snap = connect = 0;
        }
        flags ^= PAR_MSQUARES_INVERT;
        flags &= ~PAR_MSQUARES_DUAL;
        flags &= ~PAR_MSQUARES_CONNECT;
        par_msquares_meshlist* m[2];
        m[0] = par_msquares_from_function(width, height, cellsize, flags,
            context, insidefn, heightfn);
        flags ^= PAR_MSQUARES_INVERT;
        if (connect) {
            flags |= PAR_MSQUARES_CONNECT;
        }
        m[1] = par_msquares_from_function(width, height, cellsize, flags,
            context, insidefn, heightfn);
        return par_msquares_merge(m, 2, snap | connect);
    }

    int invert = flags & PAR_MSQUARES_INVERT;

    // Create the two code tables if we haven't already.  These are tables of
    // fixed constants, so it's embarassing that we use dynamic memory
    // allocation for them.  However it's easy and it's one-time-only.

    if (!point_table) {
        par_init_tables();
    }

    // Allocate the meshlist and the first mesh.

    par_msquares_meshlist* mlist = (par_msquares_meshlist*)malloc(sizeof(par_msquares_meshlist));
    mlist->nmeshes = 1;
    mlist->meshes = (par_msquares_mesh**)malloc(sizeof(par_msquares_mesh*));
    mlist->meshes[0] = (par_msquares_mesh*)malloc(sizeof(par_msquares_mesh));
    par_msquares_mesh* mesh = mlist->meshes[0];
    mesh->dim = (flags & PAR_MSQUARES_HEIGHTS) ? 3 : 2;
    int ncols = width / cellsize;
    int nrows = height / cellsize;

    // Worst case is four triangles and six verts per cell, so allocate that
    // much.

    int maxtris = ncols * nrows * 4;
    int maxpts = ncols * nrows * 6;
    int maxedges = ncols * nrows * 2;

    // However, if we include extrusion triangles for boundary edges,
    // we need space for another 4 triangles and 4 points per cell.

    uint16_t* conntris = 0;
    int nconntris = 0;
    uint16_t* edgemap = 0;
    if (flags & PAR_MSQUARES_CONNECT) {
        conntris = (uint16_t*)malloc(maxedges * 6 * sizeof(uint16_t));
        maxtris +=  maxedges * 2;
        maxpts += maxedges * 2;
        edgemap = (uint16_t*)malloc(maxpts * sizeof(uint16_t));
        for (int i = 0; i < maxpts; i++) {
            edgemap[i] = 0xffff;
        }
    }

    uint16_t* tris = (uint16_t*)malloc(maxtris * 3 * sizeof(uint16_t));
    int ntris = 0;
    float* pts = (float*)malloc(maxpts * mesh->dim * sizeof(float));
    int npts = 0;

    // The "verts" x/y/z arrays are the 4 corners and 4 midpoints around the
    // square, in counter-clockwise order.  The origin of "triangle space" is at
    // the lower-left, although we expect the image data to be in raster order
    // (starts at top-left).

    float normalization = 1.0f / MAX(width, height);
    float normalized_cellsize = cellsize * normalization;
    int maxrow = (height - 1) * width;
    uint16_t* ptris = tris;
    uint16_t* pconntris = conntris;
    float* ppts = pts;
    float vertsx[8], vertsy[8];
    uint8_t* prevrowmasks = (uint8_t*)calloc(ncols, 1);
    int* prevrowinds = (int*)calloc(sizeof(int) * ncols * 3, 1);

    // If simplification is enabled, we need to track all 'F' cells and their
    // repsective triangle indices.
    uint8_t* simplification_codes = 0;
    uint16_t* simplification_tris = 0;
    uint8_t* simplification_ntris = 0;
    if (flags & PAR_MSQUARES_SIMPLIFY) {
        simplification_codes = (uint8_t*)malloc(nrows * ncols);
        simplification_tris = (uint16_t*)malloc(nrows * ncols * sizeof(uint16_t));
        simplification_ntris = (uint8_t*)malloc(nrows * ncols);
    }

    // Do the march!
    for (int row = 0; row < nrows; row++) {
        vertsx[0] = vertsx[6] = vertsx[7] = 0;
        vertsx[1] = vertsx[5] = 0.5 * normalized_cellsize;
        vertsx[2] = vertsx[3] = vertsx[4] = normalized_cellsize;
        vertsy[0] = vertsy[1] = vertsy[2] = normalized_cellsize * (row + 1);
        vertsy[4] = vertsy[5] = vertsy[6] = normalized_cellsize * row;
        vertsy[3] = vertsy[7] = normalized_cellsize * (row + 0.5);

        int northi = row * cellsize * width;
        int southi = MIN(northi + cellsize * width, maxrow);
        int northwest = invert ^ insidefn(northi, context);
        int southwest = invert ^ insidefn(southi, context);
        int previnds[8] = {0};
        uint8_t prevmask = 0;

        for (int col = 0; col < ncols; col++) {
            northi += cellsize;
            southi += cellsize;
            if (col == ncols - 1) {
                northi--;
                southi--;
            }

            int northeast = invert ^ insidefn(northi, context);
            int southeast = invert ^ insidefn(southi, context);
            int code = southwest | (southeast << 1) | (northwest << 2) |
                (northeast << 3);

            int const* pointspec = point_table[code];
            int ptspeclength = *pointspec++;
            int currinds[8] = {0};
            uint8_t mask = 0;
            uint8_t prevrowmask = prevrowmasks[col];
            while (ptspeclength--) {
                int midp = *pointspec++;
                int bit = 1 << midp;
                mask |= bit;

                // The following six conditionals perform welding to reduce the
                // number of vertices.  The first three perform welding with the
                // cell to the west; the latter three perform welding with the
                // cell to the north.

                if (bit == 1 && (prevmask & 4)) {
                    currinds[midp] = previnds[2];
                    continue;
                }
                if (bit == 128 && (prevmask & 8)) {
                    currinds[midp] = previnds[3];
                    continue;
                }
                if (bit == 64 && (prevmask & 16)) {
                    currinds[midp] = previnds[4];
                    continue;
                }
                if (bit == 16 && (prevrowmask & 4)) {
                    currinds[midp] = prevrowinds[col * 3 + 2];
                    continue;
                }
                if (bit == 32 && (prevrowmask & 2)) {
                    currinds[midp] = prevrowinds[col * 3 + 1];
                    continue;
                }
                if (bit == 64 && (prevrowmask & 1)) {
                    currinds[midp] = prevrowinds[col * 3 + 0];
                    continue;
                }

                ppts[0] = vertsx[midp];
                ppts[1] = vertsy[midp];

                // Adjust the midpoints to a more exact crossing point.
                if (midp == 1) {
                    int begin = southi - cellsize / 2;
                    int previous = 0;
                    for (int i = 0; i < cellsize; i++) {
                        int offset = begin + i / 2 * ((i % 2) ? -1 : 1);
                        int inside = insidefn(offset, context);
                        if (i > 0 && inside != previous) {
                            ppts[0] = normalization *
                                (col * cellsize + offset - southi + cellsize);
                            break;
                        }
                        previous = inside;
                    }
                } else if (midp == 5) {
                    int begin = northi - cellsize / 2;
                    int previous = 0;
                    for (int i = 0; i < cellsize; i++) {
                        int offset = begin + i / 2 * ((i % 2) ? -1 : 1);
                        int inside = insidefn(offset, context);
                        if (i > 0 && inside != previous) {
                            ppts[0] = normalization *
                                (col * cellsize + offset - northi + cellsize);
                            break;
                        }
                        previous = inside;
                    }
                } else if (midp == 3) {
                    int begin = northi + width * cellsize / 2;
                    int previous = 0;
                    for (int i = 0; i < cellsize; i++) {
                        int offset = begin +
                            width * (i / 2 * ((i % 2) ? -1 : 1));
                        int inside = insidefn(offset, context);
                        if (i > 0 && inside != previous) {
                            ppts[1] = normalization *
                                (row * cellsize +
                                (offset - northi) / (float) width);
                            break;
                        }
                        previous = inside;
                    }
                } else if (midp == 7) {
                    int begin = northi + width * cellsize / 2 - cellsize;
                    int previous = 0;
                    for (int i = 0; i < cellsize; i++) {
                        int offset = begin +
                            width * (i / 2 * ((i % 2) ? -1 : 1));
                        int inside = insidefn(offset, context);
                        if (i > 0 && inside != previous) {
                            ppts[1] = normalization *
                                (row * cellsize +
                                (offset - northi - cellsize) / (float) width);
                            break;
                        }
                        previous = inside;
                    }
                }

                if (mesh->dim == 3) {
                    ppts[2] = heightfn(ppts[0], ppts[1], context);
                }

                ppts += mesh->dim;
                currinds[midp] = npts++;
            }

            int const* trianglespec = triangle_table[code];
            int trispeclength = *trianglespec++;

            if (flags & PAR_MSQUARES_SIMPLIFY) {
                simplification_codes[ncols * row + col] = code;
                simplification_tris[ncols * row + col] = ntris;
                simplification_ntris[ncols * row + col] = trispeclength;
            }

            // Add triangles.
            while (trispeclength--) {
                int a = *trianglespec++;
                int b = *trianglespec++;
                int c = *trianglespec++;
                *ptris++ = currinds[c];
                *ptris++ = currinds[b];
                *ptris++ = currinds[a];
                ntris++;
            }

            // Create two extrusion triangles for each boundary edge.
            if (flags & PAR_MSQUARES_CONNECT) {
                trianglespec = triangle_table[code];
                trispeclength = *trianglespec++;
                while (trispeclength--) {
                    int a = *trianglespec++;
                    int b = *trianglespec++;
                    int c = *trianglespec++;
                    int i = currinds[a];
                    int j = currinds[b];
                    int k = currinds[c];
                    int u = 0, v = 0, w = 0;
                    if ((a % 2) && (b % 2)) {
                        u = v = 1;
                    } else if ((a % 2) && (c % 2)) {
                        u = w = 1;
                    } else if ((b % 2) && (c % 2)) {
                        v = w = 1;
                    } else {
                        continue;
                    }
                    if (u && edgemap[i] == 0xffff) {
                        for (int d = 0; d < mesh->dim; d++) {
                            *ppts++ = pts[i * mesh->dim + d];
                        }
                        edgemap[i] = npts++;
                    }
                    if (v && edgemap[j] == 0xffff) {
                        for (int d = 0; d < mesh->dim; d++) {
                            *ppts++ = pts[j * mesh->dim + d];
                        }
                        edgemap[j] = npts++;
                    }
                    if (w && edgemap[k] == 0xffff) {
                        for (int d = 0; d < mesh->dim; d++) {
                            *ppts++ = pts[k * mesh->dim + d];
                        }
                        edgemap[k] = npts++;
                    }
                    if ((a % 2) && (b % 2)) {
                        *pconntris++ = i;
                        *pconntris++ = j;
                        *pconntris++ = edgemap[j];
                        *pconntris++ = edgemap[j];
                        *pconntris++ = edgemap[i];
                        *pconntris++ = i;
                    } else if ((a % 2) && (c % 2)) {
                        *pconntris++ = edgemap[k];
                        *pconntris++ = k;
                        *pconntris++ = i;
                        *pconntris++ = edgemap[i];
                        *pconntris++ = edgemap[k];
                        *pconntris++ = i;
                    } else if ((b % 2) && (c % 2)) {
                        *pconntris++ = j;
                        *pconntris++ = k;
                        *pconntris++ = edgemap[k];
                        *pconntris++ = edgemap[k];
                        *pconntris++ = edgemap[j];
                        *pconntris++ = j;
                    }
                    nconntris += 2;
                }
            }

            // Prepare for the next cell.
            prevrowmasks[col] = mask;
            prevrowinds[col * 3 + 0] = currinds[0];
            prevrowinds[col * 3 + 1] = currinds[1];
            prevrowinds[col * 3 + 2] = currinds[2];
            prevmask = mask;
            northwest = northeast;
            southwest = southeast;
            for (int i = 0; i < 8; i++) {
                previnds[i] = currinds[i];
                vertsx[i] += normalized_cellsize;
            }
        }
    }
    free(edgemap);
    free(prevrowmasks);
    free(prevrowinds);

    // Perform quick-n-dirty simplification by iterating two rows at a time.
    // In no way does this create the simplest possible mesh, but at least it's
    // fast and easy.
    if (flags & PAR_MSQUARES_SIMPLIFY) {
        int in_run = 0, start_run;

        // First figure out how many triangles we can eliminate.
        int neliminated_triangles = 0;
        for (int row = 0; row < nrows - 1; row += 2) {
            for (int col = 0; col < ncols; col++) {
                int a = simplification_codes[ncols * row + col] == 0xf;
                int b = simplification_codes[ncols * row + col + ncols] == 0xf;
                if (a && b) {
                    if (!in_run) {
                        in_run = 1;
                        start_run = col;
                    }
                    continue;
                }
                if (in_run) {
                    in_run = 0;
                    int run_width = col - start_run;
                    neliminated_triangles += run_width * 4 - 2;
                }
            }
            if (in_run) {
                in_run = 0;
                int run_width = ncols - start_run;
                neliminated_triangles += run_width * 4 - 2;
            }
        }

        // Build a new index array cell-by-cell.  If any given cell is 'F' and
        // its neighbor to the south is also 'F', then it's part of a run.
        int nnewtris = ntris + nconntris - neliminated_triangles;
        uint16_t* newtris = (uint16_t*)malloc(nnewtris * 3 * sizeof(uint16_t));
        uint16_t* pnewtris = newtris;
        in_run = 0;
        for (int row = 0; row < nrows - 1; row += 2) {
            for (int col = 0; col < ncols; col++) {
                int cell = ncols * row + col;
                int south = cell + ncols;
                int a = simplification_codes[cell] == 0xf;
                int b = simplification_codes[south] == 0xf;
                if (a && b) {
                    if (!in_run) {
                        in_run = 1;
                        start_run = col;
                    }
                    continue;
                }
                if (in_run) {
                    in_run = 0;
                    int nw_cell = ncols * row + start_run;
                    int ne_cell = ncols * row + col - 1;
                    int sw_cell = nw_cell + ncols;
                    int se_cell = ne_cell + ncols;
                    int nw_tri = simplification_tris[nw_cell];
                    int ne_tri = simplification_tris[ne_cell];
                    int sw_tri = simplification_tris[sw_cell];
                    int se_tri = simplification_tris[se_cell];
                    int nw_corner = nw_tri * 3 + 4;
                    int ne_corner = ne_tri * 3 + 0;
                    int sw_corner = sw_tri * 3 + 2;
                    int se_corner = se_tri * 3 + 1;
                    *pnewtris++ = tris[se_corner];
                    *pnewtris++ = tris[sw_corner];
                    *pnewtris++ = tris[nw_corner];
                    *pnewtris++ = tris[nw_corner];
                    *pnewtris++ = tris[ne_corner];
                    *pnewtris++ = tris[se_corner];
                }
                int ncelltris = simplification_ntris[cell];
                int celltri = simplification_tris[cell];
                for (int t = 0; t < ncelltris; t++, celltri++) {
                    *pnewtris++ = tris[celltri * 3];
                    *pnewtris++ = tris[celltri * 3 + 1];
                    *pnewtris++ = tris[celltri * 3 + 2];
                }
                ncelltris = simplification_ntris[south];
                celltri = simplification_tris[south];
                for (int t = 0; t < ncelltris; t++, celltri++) {
                    *pnewtris++ = tris[celltri * 3];
                    *pnewtris++ = tris[celltri * 3 + 1];
                    *pnewtris++ = tris[celltri * 3 + 2];
                }
            }
            if (in_run) {
                in_run = 0;
                    int nw_cell = ncols * row + start_run;
                    int ne_cell = ncols * row + ncols - 1;
                    int sw_cell = nw_cell + ncols;
                    int se_cell = ne_cell + ncols;
                    int nw_tri = simplification_tris[nw_cell];
                    int ne_tri = simplification_tris[ne_cell];
                    int sw_tri = simplification_tris[sw_cell];
                    int se_tri = simplification_tris[se_cell];
                    int nw_corner = nw_tri * 3 + 4;
                    int ne_corner = ne_tri * 3 + 0;
                    int sw_corner = sw_tri * 3 + 2;
                    int se_corner = se_tri * 3 + 1;
                    *pnewtris++ = tris[se_corner];
                    *pnewtris++ = tris[sw_corner];
                    *pnewtris++ = tris[nw_corner];
                    *pnewtris++ = tris[nw_corner];
                    *pnewtris++ = tris[ne_corner];
                    *pnewtris++ = tris[se_corner];
            }
        }
        ptris = pnewtris;
        ntris -= neliminated_triangles;
        free(tris);
        tris = newtris;
        free(simplification_codes);
        free(simplification_tris);
        free(simplification_ntris);
    }

    // Append all extrusion triangles to the main triangle array.
    // We need them to be last so that they form a contiguous sequence.
    pconntris = conntris;
    for (int i = 0; i < nconntris; i++) {
        *ptris++ = *pconntris++;
        *ptris++ = *pconntris++;
        *ptris++ = *pconntris++;
        ntris++;
    }
    free(conntris);

    // Final cleanup and return.
    assert(npts <= maxpts);
    assert(ntris <= maxtris);
    mesh->npoints = npts;
    mesh->points = pts;
    mesh->ntriangles = ntris;
    mesh->triangles = tris;
    mesh->nconntriangles = nconntris;
    return mlist;
}

#undef MIN
#undef MAX
#undef CLAMP
#endif
