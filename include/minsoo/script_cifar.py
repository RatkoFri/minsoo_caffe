
def file_gen():
    filename = "run_all_cifar.sh"
    f = open(filename,'w')
    f.write("#!/bin/bash\n\n")
    f.write("export CAFFE_ROOT=$(pwd)\n")
    f.write("rm -rf Cifar_log\n")
    f.write("mkdir Cifar_log\n")
    f.write("mkdir Cifar_log/intbit\n")
    f.write("mkdir Cifar_log/fracbit\n")
    for i in range(5,16):
        fracbits = i;
        intbits = 32 - i;
        f.write("cp ./include/minsoo/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./test_cifar.sh 2> Cifar_log/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")
        
        intbits = i;
        fracbits = 32 - i;
        f.write("cp ./include/minsoo/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./test_cifar.sh 2> Cifar_log/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")

file_gen()
