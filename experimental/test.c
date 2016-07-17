#define NANORT_C_IMPLEMENTATION
#include "nanort-c.h"

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobj_loader_c.h"

#include "imageio.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
  size_t num_vertices;
  size_t num_faces;
  float *vertices;            /* [xyz] * num_vertices */
  float *facevarying_normals; /* [xyz] * 3(triangle) * num_faces */
  float *facevarying_uvs;     /* [xyz] * 3(triangle) * num_faces */
  unsigned int *faces;        /* triangle x num_faces */
  unsigned int *material_ids; /* index x num_faces */
} Mesh;

static unsigned char ftouc(float f) {
  int i = (int)(f * 256.0f);
  if (i < 0) i = 0;
  if (i > 255) i = 255;

  return (unsigned char)(i);
}

static const char *mmap_file(size_t *len, const char *filename) {
#ifdef _WIN64
  HANDLE file =
      CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
  assert(file != INVALID_HANDLE_VALUE);

  HANDLE fileMapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
  assert(fileMapping != INVALID_HANDLE_VALUE);

  LPVOID fileMapView = MapViewOfFile(fileMapping, FILE_MAP_READ, 0, 0, 0);
  auto fileMapViewChar = (const char *)fileMapView;
  assert(fileMapView != NULL);
#else

  FILE *f;
  long file_size;
  struct stat sb;
  char *p;
  int fd;

  (*len) = 0;

  f = fopen(filename, "r");
  fseek(f, 0, SEEK_END);
  file_size = ftell(f);
  fclose(f);

  fd = open(filename, O_RDONLY);
  if (fd == -1) {
    perror("open");
    return NULL;
  }

  if (fstat(fd, &sb) == -1) {
    perror("fstat");
    return NULL;
  }

  if (!S_ISREG(sb.st_mode)) {
    fprintf(stderr, "%s is not a file\n", "lineitem.tbl");
    return NULL;
  }

  p = (char *)mmap(0, (size_t)file_size, PROT_READ, MAP_SHARED, fd, 0);

  if (p == MAP_FAILED) {
    perror("mmap");
    return NULL;
  }

  if (close(fd) == -1) {
    perror("close");
    return NULL;
  }

  (*len) = (size_t)file_size;

  return p;

#endif
}

static void VNormalize(float v[3]) {
  const float len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (len2 > 0.0f) {
    float len = (float)sqrt((double)len2);

    v[0] /= len;
    v[1] /= len;
    v[2] /= len;
  }
}

static void CalcNormal(float N[3], float v0[3], float v1[3], float v2[3]) {
  float v10[3];
  float v20[3];
  float len2;

  v10[0] = v1[0] - v0[0];
  v10[1] = v1[1] - v0[1];
  v10[2] = v1[2] - v0[2];

  v20[0] = v2[0] - v0[0];
  v20[1] = v2[1] - v0[1];
  v20[2] = v2[2] - v0[2];

  N[0] = v20[1] * v10[2] - v20[2] * v10[1];
  N[1] = v20[2] * v10[0] - v20[0] * v10[2];
  N[2] = v20[0] * v10[1] - v20[1] * v10[0];

  len2 = N[0] * N[0] + N[1] * N[1] + N[2] * N[2];
  if (len2 > 0.0f) {
    float len = (float)sqrt((double)len2);

    N[0] /= len;
    N[1] /= len;
    N[2] /= len;
  }
}

static int LoadObj(Mesh *mesh, const char *data, size_t data_len, float scale) {
  tinyobj_attrib_t attrib;
  tinyobj_shape_t *shapes;
  size_t num_shapes;

  tinyobj_material_t *materials;
  size_t num_materials;

  int ret;

  size_t num_faces;
  size_t num_vertices;

  ret = tinyobj_parse_obj(&attrib, &shapes, &num_shapes, &materials,
                          &num_materials, data, data_len,
                          TINYOBJ_FLAG_TRIANGULATE);

  if (ret != TINYOBJ_SUCCESS) {
    fprintf(stderr, "faield to parse .obj\n");
    return -1;
  }

  printf("# of shapes    : %d\n", (int)num_shapes);
  printf("# of materials : %d\n", (int)num_materials);

  /* Assume all triangle geometry */
  {
    num_faces = attrib.num_face_num_verts;
    num_vertices = attrib.num_vertices;

    mesh->num_faces = num_faces;
    mesh->num_vertices = num_vertices;
    mesh->vertices = (float *)malloc(sizeof(float) * num_vertices * 3);
    mesh->faces = (unsigned int *)malloc(sizeof(unsigned int) * num_faces * 3);
    mesh->material_ids =
        (unsigned int *)malloc(sizeof(unsigned int) * num_faces);
    memset(mesh->material_ids, 0, sizeof(unsigned int) * num_faces);
    mesh->facevarying_normals =
        (float *)malloc(sizeof(float) * num_faces * 3 * 3);
    mesh->facevarying_uvs = (float *)malloc(sizeof(float) * num_faces * 3 * 2);
    memset(mesh->facevarying_uvs, 0, sizeof(float) * 2 * 3 * num_faces);
  }

  {
    size_t i;
    size_t f;

    for (i = 0; i < attrib.num_vertices; i++) {
      mesh->vertices[3 * i + 0] = scale * attrib.vertices[3 * i + 0];
      mesh->vertices[3 * i + 1] = scale * attrib.vertices[3 * i + 1];
      mesh->vertices[3 * i + 2] = scale * attrib.vertices[3 * i + 2];
    }

    for (f = 0; f < attrib.num_face_num_verts; f++) {
      /* Assume all triangle geometry */
      mesh->faces[3 * f + 0] = (unsigned int)attrib.faces[3 * f + 0].v_idx;
      mesh->faces[3 * f + 1] = (unsigned int)attrib.faces[3 * f + 1].v_idx;
      mesh->faces[3 * f + 2] = (unsigned int)attrib.faces[3 * f + 2].v_idx;
    }

    if (attrib.num_normals > 0) {
      for (f = 0; f < attrib.num_face_num_verts; f++) {
        int i0 = attrib.faces[3 * f + 0].vn_idx;
        int i1 = attrib.faces[3 * f + 1].vn_idx;
        int i2 = attrib.faces[3 * f + 2].vn_idx;

        if (i0 >= 0 && i1 >= 0 && i2 >= 0) {
          float n0[3];
          float n1[3];
          float n2[3];
          size_t k;

          for (k = 0; k < 3; k++) {
            n0[k] = attrib.normals[3 * (size_t)i0 + k];
            n1[k] = attrib.normals[3 * (size_t)i1 + k];
            n2[k] = attrib.normals[3 * (size_t)i2 + k];
          }

          for (k = 0; k < 3; k++) {
            mesh->facevarying_normals[9 * f + 0 + k] = n0[k];
            mesh->facevarying_normals[9 * f + 3 + k] = n1[k];
            mesh->facevarying_normals[9 * f + 6 + k] = n2[k];
          }
        } else { /* invalid normal */
          int f0, f1, f2;
          float v0[3], v1[3], v2[3];
          float N[3];

          f0 = attrib.faces[3 * f + 0].v_idx;
          f1 = attrib.faces[3 * f + 1].v_idx;
          f2 = attrib.faces[3 * f + 2].v_idx;

          v0[0] = attrib.vertices[3 * f0 + 0];
          v0[1] = attrib.vertices[3 * f0 + 1];
          v0[2] = attrib.vertices[3 * f0 + 2];

          v1[0] = attrib.vertices[3 * f1 + 0];
          v1[1] = attrib.vertices[3 * f1 + 1];
          v1[2] = attrib.vertices[3 * f1 + 2];

          v2[0] = attrib.vertices[3 * f2 + 0];
          v2[1] = attrib.vertices[3 * f2 + 1];
          v2[2] = attrib.vertices[3 * f2 + 2];

          CalcNormal(N, v0, v1, v2);

          mesh->facevarying_normals[3 * (3 * f + 0) + 0] = N[0];
          mesh->facevarying_normals[3 * (3 * f + 0) + 1] = N[1];
          mesh->facevarying_normals[3 * (3 * f + 0) + 2] = N[2];

          mesh->facevarying_normals[3 * (3 * f + 1) + 0] = N[0];
          mesh->facevarying_normals[3 * (3 * f + 1) + 1] = N[1];
          mesh->facevarying_normals[3 * (3 * f + 1) + 2] = N[2];

          mesh->facevarying_normals[3 * (3 * f + 2) + 0] = N[0];
          mesh->facevarying_normals[3 * (3 * f + 2) + 1] = N[1];
          mesh->facevarying_normals[3 * (3 * f + 2) + 2] = N[2];
        }
      }
    } else {
      /* calc geometric normal */
      for (f = 0; f < attrib.num_face_num_verts; f++) {
        int f0, f1, f2;
        float v0[3], v1[3], v2[3];
        float N[3];

        f0 = attrib.faces[3 * f + 0].v_idx;
        f1 = attrib.faces[3 * f + 1].v_idx;
        f2 = attrib.faces[3 * f + 2].v_idx;

        v0[0] = attrib.vertices[3 * f0 + 0];
        v0[1] = attrib.vertices[3 * f0 + 1];
        v0[2] = attrib.vertices[3 * f0 + 2];

        v1[0] = attrib.vertices[3 * f1 + 0];
        v1[1] = attrib.vertices[3 * f1 + 1];
        v1[2] = attrib.vertices[3 * f1 + 2];

        v2[0] = attrib.vertices[3 * f2 + 0];
        v2[1] = attrib.vertices[3 * f2 + 1];
        v2[2] = attrib.vertices[3 * f2 + 2];

        CalcNormal(N, v0, v1, v2);

        mesh->facevarying_normals[3 * (3 * f + 0) + 0] = N[0];
        mesh->facevarying_normals[3 * (3 * f + 0) + 1] = N[1];
        mesh->facevarying_normals[3 * (3 * f + 0) + 2] = N[2];

        mesh->facevarying_normals[3 * (3 * f + 1) + 0] = N[0];
        mesh->facevarying_normals[3 * (3 * f + 1) + 1] = N[1];
        mesh->facevarying_normals[3 * (3 * f + 1) + 2] = N[2];

        mesh->facevarying_normals[3 * (3 * f + 2) + 0] = N[0];
        mesh->facevarying_normals[3 * (3 * f + 2) + 1] = N[1];
        mesh->facevarying_normals[3 * (3 * f + 2) + 2] = N[2];
      }
    }

    /* @todo { texcoord, material_id, etc } */
  }

  /* free tinyobj data */
  {
    tinyobj_attrib_free(&attrib);
    tinyobj_shapes_free(shapes, num_shapes);
    tinyobj_materials_free(materials, num_materials);
  }

  return 0;
}

int main(int argc, char **argv) {
  const char *obj_filename;
  const char *obj_data;
  size_t obj_data_len;
  float scale = 1.0f;

  Mesh mesh;

  if (argc < 2) {
    printf("test input.obj <scale>\n");
    exit(-1);
  }

  if (argc > 2) {
    scale = (float)atof(argv[2]);
  }

  obj_filename = argv[1];

  obj_data = mmap_file(&obj_data_len, obj_filename);
  if (obj_data == NULL || obj_data_len == 0) {
    fprintf(stderr, "failed to map file: %s\n", obj_filename);
    exit(-1);
  }

  {
    int ret = LoadObj(&mesh, obj_data, obj_data_len, scale);
    if (ret < 0) {
      fprintf(stderr, "failed to load .obj %s\n", obj_filename);
      exit(-1);
    }
  }

  {
    nanort_bvh_accel_t accel;
    nanort_bvh_build_statistics_t stats;
    nanort_bvh_build_options_t build_options;
    int ret;
    float bmin[3], bmax[3];

    nanort_bvh_accel_init(&accel);
    nanort_bvh_build_options_init(&build_options);
    /* `stats` will be intialized inside of nanort_bvh_accel_build(), thus no
     * initialization required for `stats` here. */

    ret = nanort_bvh_accel_build(&accel, &stats, mesh.vertices,
                                 (unsigned int)mesh.num_faces, mesh.faces,
                                 &build_options);
    assert(ret == NANORT_SUCCESS);

    nanort_bvh_accel_bounding_box(bmin, bmax, &accel);

    printf("BVH statistics\n");
    printf("  bmin              : %f, %f, %f\n", (double)bmin[0],
           (double)bmin[1], (double)bmin[2]);
    printf("  bmax              : %f, %f, %f\n", (double)bmax[0],
           (double)bmax[1], (double)bmax[2]);
    printf("  max tree depth    : %d\n", stats.max_tree_depth);
    printf("  # of leaf nodes   : %d\n", stats.num_leaf_nodes);
    printf("  # of branch nodes : %d\n", stats.num_branch_nodes);

    /* very simple rendering */
    {
      int x, y;
      int width = 512;
      int height = 512;
      nanort_bvh_trace_options_t trace_options;

      unsigned char *rgb =
          (unsigned char *)malloc((size_t)(width * height * 3));
      memset(rgb, 0, (size_t)(width * height * 3));

      /* Use default trace option. */
      nanort_bvh_trace_options_init(&trace_options);

/* Shoot rays. */
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
          float dir[3];
          nanort_ray_t ray;
          float kFar = 1.0e+30f;
          int hit;
          nanort_intersection_t isect;

          /* @fixme { adjust eye position. (0, 5, 20) is set for
           * `cornellbox_suzanne.obj` } */
          ray.org[0] = 0.0f;
          ray.org[1] = 5.0f;
          ray.org[2] = 20.0f;

          dir[0] = (x / (float)width) - 0.5f;
          dir[1] = (y / (float)height) - 0.5f;
          dir[2] = -1.0f;
          VNormalize(dir);
          ray.dir[0] = dir[0];
          ray.dir[1] = dir[1];
          ray.dir[2] = dir[2];

          ray.min_t = 0.0f;
          ray.max_t = kFar;

          hit = nanort_bvh_accel_traverse(&isect, &accel, mesh.vertices,
                                          mesh.faces, &ray, &trace_options);
          if (hit) {
            /* @fixme { write your shader here. } */
            float normal[3] = {1, 1, 1};
            unsigned int fid = isect.prim_id;
            if (mesh.facevarying_normals) {
              normal[0] = mesh.facevarying_normals[9 * fid + 0];
              normal[1] = mesh.facevarying_normals[9 * fid + 1];
              normal[2] = mesh.facevarying_normals[9 * fid + 2];
            }
            /* Flip Y */
            rgb[3 * ((height - y - 1) * width + x) + 0] =
                ftouc(0.5f * normal[0] + 0.5f);
            rgb[3 * ((height - y - 1) * width + x) + 1] =
                ftouc(0.5f * normal[1] + 0.5f);
            rgb[3 * ((height - y - 1) * width + x) + 2] =
                ftouc(0.5f * normal[2] + 0.5f);
          }
        }
      }

      {
        ret = save_png("render.png", width, height, rgb);
        if (ret > 0) {
          printf("saved to render.png\n");
        }

        free(rgb);
      }
    }

    nanort_bvh_accel_free(&accel);

    {
      free(mesh.vertices);
      free(mesh.faces);
      free(mesh.material_ids);
      free(mesh.facevarying_normals);
      free(mesh.facevarying_uvs);
    }
  }

  return EXIT_SUCCESS;
}
