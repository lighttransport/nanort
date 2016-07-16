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
  float *vertices;              /* [xyz] * num_vertices */
  float *facevarying_normals;   /* [xyz] * 3(triangle) * num_faces */
  float *facevarying_uvs;       /* [xyz] * 3(triangle) * num_faces */
  unsigned int *faces;          /* triangle x num_faces */
  unsigned int *material_ids;   /* index x num_faces */
} Mesh;


static const char* mmap_file(size_t* len, const char* filename) {
#ifdef _WIN64
  HANDLE file =
      CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
  assert(file != INVALID_HANDLE_VALUE);

  HANDLE fileMapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
  assert(fileMapping != INVALID_HANDLE_VALUE);

  LPVOID fileMapView = MapViewOfFile(fileMapping, FILE_MAP_READ, 0, 0, 0);
  auto fileMapViewChar = (const char*)fileMapView;
  assert(fileMapView != NULL);
#else

  FILE* f;
  long file_size;
  struct stat sb;
  char* p;
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

  p = (char*)mmap(0, (size_t)file_size, PROT_READ, MAP_SHARED, fd, 0);

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
  }
}

#if 0
static int load_obj_and(float bmin[3], float bmax[3],
                             const char* filename) {
  tinyobj_attrib_t attrib;
  tinyobj_shape_t* shapes = NULL;
  size_t num_shapes;
  tinyobj_material_t* materials = NULL;
  size_t num_materials;

  size_t data_len = 0;
  const char* data = get_file_data(&data_len, filename);
  if (data == NULL) {
    exit(-1);
    return 0;
  }
  printf("filesize: %d\n", (int)data_len);

  {
    unsigned int flags = TINYOBJ_FLAG_TRIANGULATE;
    int ret = tinyobj_parse_obj(&attrib, &shapes, &num_shapes, &materials,
                                &num_materials, data, data_len, flags);
    if (ret != TINYOBJ_SUCCESS) {
      return 0;
    }

    printf("# of shapes    = %d\n", (int)num_shapes);
    printf("# of materiasl = %d\n", (int)num_materials);

    if (0) {
      int i;
      for (i = 0; i < num_shapes; i++) {
        printf("shape[%d] name = %s\n", i, shapes[i].name);
      }
    }

  }

  bmin[0] = bmin[1] = bmin[2] = FLT_MAX;
  bmax[0] = bmax[1] = bmax[2] = -FLT_MAX;

  {
    DrawObject o;
    float* vb;
    /* std::vector<float> vb; //  */
    size_t face_offset = 0;
    size_t i;

    /* Assume triangulated face. */
    size_t num_triangles = attrib.num_face_num_verts;
    size_t stride = 9; /* 9 = pos(3float), normal(3float), color(3float) */

    vb = (float*)malloc(sizeof(float) * stride * num_triangles * 3);

    for (i = 0; i < attrib.num_face_num_verts; i++) {
      size_t f;
      assert(attrib.face_num_verts[i] % 3 ==
             0); /* assume all triangle faces. */
      for (f = 0; f < attrib.face_num_verts[i] / 3; f++) {
        int k;
        float v[3][3];
        float n[3][3];
        float c[3];
        float len2;

        tinyobj_vertex_index_t idx0 = attrib.faces[face_offset + 3 * f + 0];
        tinyobj_vertex_index_t idx1 = attrib.faces[face_offset + 3 * f + 1];
        tinyobj_vertex_index_t idx2 = attrib.faces[face_offset + 3 * f + 2];

        for (k = 0; k < 3; k++) {
          int f0 = idx0.v_idx;
          int f1 = idx1.v_idx;
          int f2 = idx2.v_idx;
          assert(f0 >= 0);
          assert(f1 >= 0);
          assert(f2 >= 0);

          v[0][k] = attrib.vertices[3 * f0 + k];
          v[1][k] = attrib.vertices[3 * f1 + k];
          v[2][k] = attrib.vertices[3 * f2 + k];
          bmin[k] = (v[0][k] < bmin[k]) ? v[0][k] : bmin[k];
          bmin[k] = (v[1][k] < bmin[k]) ? v[1][k] : bmin[k];
          bmin[k] = (v[2][k] < bmin[k]) ? v[2][k] : bmin[k];
          bmax[k] = (v[0][k] > bmax[k]) ? v[0][k] : bmax[k];
          bmax[k] = (v[1][k] > bmax[k]) ? v[1][k] : bmax[k];
          bmax[k] = (v[2][k] > bmax[k]) ? v[2][k] : bmax[k];
        }

        if (attrib.num_normals > 0) {
          int f0 = idx0.vn_idx;
          int f1 = idx1.vn_idx;
          int f2 = idx2.vn_idx;
          if (f0 >= 0 && f1 >= 0 && f2 >= 0) {
            assert(f0 < attrib.num_normals);
            assert(f1 < attrib.num_normals);
            assert(f2 < attrib.num_normals);
            for (k = 0; k < 3; k++) {
              n[0][k] = attrib.normals[3 * f0 + k];
              n[1][k] = attrib.normals[3 * f1 + k];
              n[2][k] = attrib.normals[3 * f2 + k];
            }
          } else { /* normal index is not defined for this face */
            /* compute geometric normal */
            CalcNormal(n[0], v[0], v[1], v[2]);
            n[1][0] = n[0][0];
            n[1][1] = n[0][1];
            n[1][2] = n[0][2];
            n[2][0] = n[0][0];
            n[2][1] = n[0][1];
            n[2][2] = n[0][2];
          }
        } else {
          /* compute geometric normal */
          CalcNormal(n[0], v[0], v[1], v[2]);
          n[1][0] = n[0][0];
          n[1][1] = n[0][1];
          n[1][2] = n[0][2];
          n[2][0] = n[0][0];
          n[2][1] = n[0][1];
          n[2][2] = n[0][2];
        }

        for (k = 0; k < 3; k++) {
          vb[(3 * i + k) * stride + 0] = v[k][0];
          vb[(3 * i + k) * stride + 1] = v[k][1];
          vb[(3 * i + k) * stride + 2] = v[k][2];
          vb[(3 * i + k) * stride + 3] = n[k][0];
          vb[(3 * i + k) * stride + 4] = n[k][1];
          vb[(3 * i + k) * stride + 5] = n[k][2];

          /* Use normal as color. */
          c[0] = n[k][0];
          c[1] = n[k][1];
          c[2] = n[k][2];
          len2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
          if (len2 > 0.0f) {
            float len = (float)sqrt(len2);

            c[0] /= len;
            c[1] /= len;
            c[2] /= len;
          }

          vb[(3 * i + k) * stride + 6] = (c[0] * 0.5 + 0.5);
          vb[(3 * i + k) * stride + 7] = (c[1] * 0.5 + 0.5);
          vb[(3 * i + k) * stride + 8] = (c[2] * 0.5 + 0.5);
        }
      }
      face_offset += attrib.face_num_verts[i];
    }

    o.vb = 0;
    o.numTriangles = 0;
    if (num_triangles > 0) {
      glGenBuffers(1, &o.vb);
      glBindBuffer(GL_ARRAY_BUFFER, o.vb);
      glBufferData(GL_ARRAY_BUFFER, num_triangles * 3 * stride * sizeof(float),
                   vb, GL_STATIC_DRAW);
      o.numTriangles = num_triangles;
    }

    free(vb);

    gDrawObject = o;
  }

  printf("bmin = %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("bmax = %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  tinyobj_attrib_free(&attrib);
  tinyobj_shapes_free(shapes, num_shapes);
  tinyobj_materials_free(materials, num_materials);

  return 1;
}
#endif

static int LoadObj(Mesh *mesh, const char *data, size_t data_len, float scale) {
  tinyobj_attrib_t attrib;
  tinyobj_shape_t *shapes;
  size_t num_shapes;

  tinyobj_material_t *materials;
  size_t num_materials;

  int ret;

  size_t num_faces;
  size_t num_vertices;

  ret = tinyobj_parse_obj(&attrib, &shapes, &num_shapes, &materials, &num_materials, data, data_len, TINYOBJ_FLAG_TRIANGULATE);

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
    mesh->vertices = (float*)malloc(sizeof(float) * num_vertices * 3);
    mesh->faces = (unsigned int*)malloc(sizeof(unsigned int) * num_faces * 3);
    mesh->material_ids = (unsigned int *)malloc(sizeof(unsigned int) * num_faces);
    memset(mesh->material_ids, 0, sizeof(unsigned int) * num_faces);
    mesh->facevarying_normals = (float*)malloc(sizeof(float) * num_faces * 3 * 3);
    mesh->facevarying_uvs = (float*)malloc(sizeof(float) * num_faces * 3 * 2);
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

          f0 = attrib.faces[3*f+0].v_idx;
          f1 = attrib.faces[3*f+1].v_idx;
          f2 = attrib.faces[3*f+2].v_idx;


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

        f0 = attrib.faces[3*f+0].v_idx;
        f1 = attrib.faces[3*f+1].v_idx;
        f2 = attrib.faces[3*f+2].v_idx;


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

  return 0;
}


int
main(
	int argc,
	char **argv)
{
  const char* obj_filename;
  const char* obj_data;
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

	return EXIT_SUCCESS;
}
