# NyaZy

## Current `nyacc` behavior

Emit MLIR and converted LLVM IR from MLIR where the code is just printing *hello world*.
Try the emitted LLVM IR.
```bash
# run the nyacc to see emitted LLVM IR
$ ./bin/nyacc
# copy and paste emitted LLVM IR
$ vim tmp.ll
# run the LLVM IR using LLVM's lli
$ ./bin lli tmp.ll
```

## Setup

```bash
# build LLVM and other external libraries
$ ./bin thirdparty
# configure this project
$ ./bin configure
# build this project
$ ./bin build
```

## `bin` shell script
`/bin` can be used to invoke executables or one linear command for everyday developments.

Run `./bin help` to see all the available commands.