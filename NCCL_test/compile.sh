if [ "$1" = "" ]
then
  echo "Usage: $0 <c file to generate cfg for>"
  exit
fi

nvcc -lnccl  $1 -o ${1%%.*}.o -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/nccl/lib