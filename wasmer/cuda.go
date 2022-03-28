package wasmer

// #include <wasmer.h>
import "C"

import (
	"runtime"
	"unsafe"
)

type CudaEnvironment struct {
	_inner *C.cuda_env_t
}

// NewCudaEnvironment will instantiates a new CudaEnvironment
func NewCudaEnvironment() *CudaEnvironment {
	var env *C.cuda_env_t
	env = C.cuda_env_new()

	cudaEnv := &CudaEnvironment{
		_inner: env,
	}

	runtime.SetFinalizer(cudaEnv, func(env *CudaEnvironment) {
		C.cuda_env_delete(env.inner())
	})

	return cudaEnv
}

func (cudaEnv *CudaEnvironment) inner() *C.cuda_env_t {
	return cudaEnv._inner
}

// GenerateImportObject will generate an ImportObject for cuda env
func (cudaEnv *CudaEnvironment) GenerateImportObject(store *Store) (*ImportObject, error) {
	importObject := NewImportObject()
	err := cudaEnv.AddImportObject(store, importObject)
	return importObject, err
}

// AddImportObject will add cuda Imports to an existed ImportObject
func (cudaEnv *CudaEnvironment) AddImportObject(store *Store, importObject *ImportObject) error {
	var cudaNamedExterns C.wasmer_named_extern_vec_t

	err := maybeNewErrorFromWasmer(func() bool {
		return C.cuda_get_unordered_imports(store.inner(), cudaEnv.inner(), &cudaNamedExterns) == false
	})

	if err != nil {
		return err
	}

	externNum := int(cudaNamedExterns.size)
	firstNamedExtern := unsafe.Pointer(cudaNamedExterns.data)
	sizeOfNamedExtern := unsafe.Sizeof(firstNamedExtern)

	for i := 0; i < externNum; i++ {
		current := *(**C.wasmer_named_extern_t)(unsafe.Pointer(uintptr(firstNamedExtern) + uintptr(i)*sizeOfNamedExtern))
		module := nameToString(C.wasmer_named_extern_module(current))
		name := nameToString(C.wasmer_named_extern_name(current))
		extern := newExtern(C.wasm_extern_copy(C.wasmer_named_extern_unwrap(current)), nil)

		_, exists := importObject.externs[module]

		if exists == false {
			importObject.externs[module] = make(map[string]IntoExtern)
		}

		importObject.externs[module][name] = extern
	}

	C.wasmer_named_extern_vec_delete(&cudaNamedExterns)

	return nil
}
