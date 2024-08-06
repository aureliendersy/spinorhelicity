When generating data a few commands are helpful in combining and merging data files.
We list here some of the relevant ones.

1) The data generated is in the format of:
  Scrambled Amplitude \t Simple Amplitude & Identities used & Little Group and Mass dimension scaling.
  To instead train on just the amplitude pairs, the following command will return a file with only the information before the & sign
  ```shell
  awk -F'&' '{print $1}' data_ids.prefix> data.prefix
  ```
2) To eliminate duplicate lines in the data file we use
```shell
 cat data.prefix | awk 'NF{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' data.prefix | sort -n -r > data.prefix.counts
```
3) To split the raw prefix data into training, validation and testing set we call the relevant script (where the second argument determines the size of the validation and testing sets)
```shell
python split_data.py data.prefix.counts 5000
```
4) To verify that the input (scrambled amplitudes) are not present in the reference training set we use (example given for training set)
   ```shell
   mv data.prefix.counts.test data.prefix.counts.test.old
   awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' <(cat data.prefix.counts.train) data.prefix.counts.test.old > data.prefix.counts.test
   ```
5) When combining different training sets together we can get a large training set that is too big to load at once. To ensure that the loaded examples are relevant, but still randomly selected, we can first shuffle the training set randomly and then sort it out according to the number of duplicate examples. The random shuffle is needed here to ensure that examples which have the same number of duplicates are arranged randomly and not according to the order in which the datasets were combined.
```shell
   mv data.prefix.counts.train data.prefix.counts.train_combined
   shuf data.prefix.counts.train_combined > temp_file
   sort -t '|' -k 1,1nr temp_file > data.prefix.counts.train
   rm temp_file
   ```
6) To restrict the number of tokens that compose the input (and output amplitudes) we use
   ```shell
    python reduce_data_length.py data.prefix.counts.train 1000
   ```
