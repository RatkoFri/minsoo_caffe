mkdir Detail;
mkdir Log_col;
echo "1" > current_mult;
./test_alexnet.sh 2> float_log;
mv float_log Log_col/;
mv detail.log Detail/float.log;
echo "2" > current_mult;
./test_alexnet.sh 2> fixed_log;
mv fixed_log Log_col/;
mv detail.log Detail/fixed.log;
echo "3" > current_mult;
./test_alexnet.sh 2> mitch_log;
mv mitch_log Log_col/;
mv detail.log Detail/mitch.log;
echo "4" > current_mult;
./test_alexnet.sh 2> iterlog_log;
mv iterlog_log Log_col/;
mv detail.log Detail/iterlog.log;
echo "5" > current_mult;
echo "6" > DRUM_K;
./test_alexnet.sh 2> drum6_log;
mv drum6_log Log_col/;
mv detail.log Detail/drum6.log;
echo "6" > current_mult;
echo "5" > DRUM_K;
./test_alexnet.sh 2> mitchk5_log;
mv mitchk5_log Log_col/;
mv detail.log Detail/mitchk5.log;
echo "7" > current_mult;
echo "5" > DRUM_K;
./test_alexnet.sh 2> mitchk_bias5_log;
mv mitchk_bias5_log Log_col/;
mv detail.log Detail/mitchk_bias5.log;
echo "8" > current_mult;
echo "5" > DRUM_K;
./test_alexnet.sh 2> c1_mitchk_bias5_log;
mv c1_mitchk_bias5_log Log_col/;
mv detail.log Detail/c1_mitchk_bias5.log;
echo "9" > current_mult;
./test_alexnet.sh 2> asm_log;
mv asm_log Log_col/;
mv detail.log Detail/asm.log;
