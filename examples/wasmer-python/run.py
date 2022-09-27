# main.py
from wasmer import engine, wasi, ImportObject, Function, Store, Memory, MemoryType, Module, Instance, Float32Array
from wasmer_compiler_cranelift import Compiler

class RuntimeHelper:
    def __init__(self, store):
        self.store = store
        self.memory = None

    def set_memory(self, memory):
        self.memory = memory

    def host_functions(self):

        import_object = ImportObject()

        def myabort():
            pass

        def mymemcpy_big(dest: 'i32', src: 'i32', n: 'i32'):
            print("memcpy big: dest {}, src {}, n {}".format(dest, src, n))
            buf = bytearray(self.memory.buffer)
            buf[dest:dest+n] = buf[src:src+n]

        def myresize_heap(a: 'i32') -> int:
            print("resize_heap: a {}".format(a))
            raise "TODO"

        def myfd_write(a: 'i32', b: 'i32', c: 'i32', d: 'i32') -> int:
            print("fd_write")

        abort_host_function = Function(self.store, myabort)
        memcpy_big_host_function = Function(self.store, mymemcpy_big)
        resize_heap_host_function = Function(self.store, myresize_heap)

        fd_write_host_function = Function(self.store, myfd_write)

        # Now let's register the `sum` import inside the `env` namespace.
        import_object.register(
            "env",
            {
                "abort": abort_host_function,
                "emscripten_memcpy_big": memcpy_big_host_function,
                "emscripten_resize_heap": resize_heap_host_function,
            },
        )

        import_object.register(
            "wasi_snapshot_preview1",
            {
                "fd_write": fd_write_host_function,
            }
        )

        return import_object

# Let's define the store, that holds the engine, that holds the compiler.
store = Store(engine.Universal(Compiler))

wasm_bytes = open('a.out.wasm', 'rb').read()

# Let's compile the module to be able to execute it!
#module = Module(store, open('a.out.wasm', 'rb').read())
module = Module(store, wasm_bytes)
print(module)

wasi_version = wasi.get_version(module, strict=True)
print("WASI version", wasi_version)

#import_object = wasi_env.generate_import_object(store, wasi_version)

helper = RuntimeHelper(store)

import_object = helper.host_functions()

# Now the module is compiled, we can instantiate it.
instance = Instance(module, import_object)

m = instance.exports.memory
helper.set_memory(m)

#input_pointer = instance.exports.allocate(4 * 4)
#print(input_pointer)

#print(m.size)
#m.grow(2)
#print(m.size)

bt = bytearray(m.buffer)
f32_view = m.float32_view(0)
f32_view[0] = 1.0
f32_view[1] = 2.0
f32_view[2] = 3.3
f32_view[3] = 4.4
print(f32_view)

ret = instance.exports.func(0, 4, 16)
print(ret)

#f32_view = m.float32_view(0)

print("out", f32_view[4])
#print(instance.exports)
