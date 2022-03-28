package main

import (
	"fmt"
	"github.com/yanghaku/wasmer-gpu-go/wasmer"
	"io/ioutil"
	"os"
)

func checkErr(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
func run(wasmName string, runArg string) {
	wasmBytes, err := ioutil.ReadFile(wasmName)
	checkErr(err)

	engine := wasmer.NewEngine()
	store := wasmer.NewStore(engine)

	module, err := wasmer.NewModule(store, wasmBytes)
	checkErr(err)

	wasiEnv, err := wasmer.NewWasiStateBuilder(wasmName).Argument(runArg).Finalize()
	checkErr(err)

	importObject, err := wasiEnv.GenerateImportObject(store, module)
	checkErr(err)

	cudaEnv := wasmer.NewCudaEnvironment()
	err = cudaEnv.AddImportObject(store, importObject)
	checkErr(err)

	instance, err := wasmer.NewInstance(module, importObject)
	checkErr(err)

	start, err := instance.Exports.GetWasiStartFunction()
	checkErr(err)

	_, err = start()
	checkErr(err)
}

func main() {
	run("device.wasm", "")
	run("mem.wasm", "")
	run("sumArray.wasm", "")
	run("matrixMulCuda.wasm", "1024")
}
