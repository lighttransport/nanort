# NanoSG

Simple, minimal and header-only scene graph library for NanoRT.

## Data structure

### Node

Node represents scene graph node. Tansformation node or Mesh(shape) node.
Node is interpreted as transformation node when passing `nullptr` to Node class constructure.

Node can contain multiple children.

### Scene

Scene contains root nodes and provides the method to find an intersection of nodes.

## User defined data structure

Following are required in user application.

### Mesh class

Current example code assumes mesh is all composed of triangle meshes.

Following method must be implemented for `Scene::Traversal`.

```
///
/// Get the geometric normal and the shading normal at `face_idx' th face.
///
template<typename T>
void GetNormal(T Ng[3], T Ns[3], const unsigned int face_idx, const T u, const T v) const;
```

### Intersection class

Represents intersection(hit) information.

## Memory management

`Scene` and `Node` does not create a copy of asset data(e.g. vertices, indices). Thus user must care about memory management of scene assets in user side.

## API

API is still subject to change.

### Node

```
void Node::SetName(const std::string &name);
```

Set (unique) name for the node.

```
void Node::AddChild(const type &child);
```

Add node as child node.

```
void Node::SetLocalXform(const T xform[4][4]) {
```

Set local transformation matrix.

### Scene

```
bool Scene::AddNode(const Node<T, M> &node);
```

Add a node to the scene.

```
bool Scene::Commit() {
```

Commit the scene. After adding nodes to the scene or changed transformation matrix, call this `Commit` before tracing rays.
`Commit` triggers BVH build in each nodes and updates node's transformation matrix.

```
template<class H>
bool Scene::Traverse(nanort::Ray<T> &ray, H *isect, const bool cull_back_face = false) const;
```

Trace ray into the scene and find an intersection.
Returns `true` when there is an intersection and hit information is stored in `isect`.
