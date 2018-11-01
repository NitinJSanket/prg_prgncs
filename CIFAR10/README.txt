For Compilation:
mvNCCompile inference.meta -s 12 -in Input -on prLogitsT/BiasAdd -o inference.graph

For Profiling:
mvNCProfile inference.meta -s 12 -in Input -on prLogitsT/BiasAdd -is 32 32

For Checking:
mvNCCheck inference.meta -s 12 -in=Input -on=prLogitsT/BiasAdd -is 32 32 -cs 0,1,2

