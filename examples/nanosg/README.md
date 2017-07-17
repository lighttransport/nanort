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

* Mesh class
  * Current example code assumes mesh is all composed of triangle meshes.
* Intersection class
  * Represents intersection(hit) information.

## Memory management

`Scene` and `Node` does not create a copy of asset data(e.g. vertices, indices). Thus user must care about memory management of scene assets in user side.

## API

```
Node::AddChild(const type &child);
```

Add node as child node.

```
Node::AddChild(const type &child);
```

Add node as child node.
