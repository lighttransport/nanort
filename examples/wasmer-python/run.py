# main.py
from wasmer import engine, wasi, ImportObject, Function, Store, Memory, MemoryType, Module, Instance, Float32Array
from wasmer_compiler_cranelift import Compiler

def host_functions(store):

    import_object = ImportObject()

    def myabort():
        pass

    def mymemcpy_big(dest: 'i32', src: 'i32', n: 'i32'):
        print("memcpy big")
        pass

    def myresize_heap(a: 'i32') -> int:
        print("resize_heap")
        pass

    abort_host_function = Function(store, myabort)
    memcpy_big_host_function = Function(store, mymemcpy_big)
    resize_heap_host_function = Function(store, myresize_heap)

    # Now let's register the `sum` import inside the `env` namespace.
    import_object.register(
        "env",
        {
            "abort": abort_host_function,
            "emscripten_memcpy_big": memcpy_big_host_function,
            "emscripten_resize_heap": resize_heap_host_function,
        },
    )

    return import_object

# Let's define the store, that holds the engine, that holds the compiler.
store = Store(engine.Universal(Compiler))

wasm_bytes = open('a.out.wasm', 'rb').read()

# Let's compile the module to be able to execute it!
#module = Module(store, open('a.out.wasm', 'rb').read())
module = Module(store, wasm_bytes)
print(module)

#wasi_version = wasi.get_version(module, strict=True)
#print("WASI version", wasi_version)

#import_object = wasi_env.generate_import_object(store, wasi_version)

import_object = host_functions(store)

# Now the module is compiled, we can instantiate it.
instance = Instance(module, import_object)

#input_pointer = instance.exports.allocate(4 * 4)
#print(input_pointer)

m = instance.exports.memory
print(m.size)
m.grow(2)
print(m.size)

f32_view = m.float32_view(1)
f32_view[0] = 1.0
f32_view[1] = 2.0
f32_view[2] = 3.3
f32_view[3] = 4.4
print(f32_view)

instance.exports.func
print(instance.exports)
