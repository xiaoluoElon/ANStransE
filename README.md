# ANStransE 
## Dependencies
- Python 3.6+
- [PyTorch](http://pytorch.org/) 1.0+

## Results
The results of **ANStransE** and the baseline model **ModE** on **WN18RR**, **FB15k-237** and **YAGO3-10** are as follows.
 
### WN18RR
| | MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ANStransE | 0.329 | 0.230 | 0.366 | 0.527 |

### FB15k-237
| | MRR | HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ANStransE | 0.229 |  0.019 | 0.406 | 0.535 |

### YAGO3-10
| | MRR | HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ANStransE | 0.505 |  0.406 | 0.568 | 0.683 |



## Running the code 

### Usage
```
bash runs.sh {train | valid | test} {ModE | HAKE} {wn18rr | FB15k-237 | YAGO3-10} <gpu_id> \
<save_id> <train_batch_size> <negative_sample_size> <hidden_dim> <gamma> <alpha> \
<learning_rate> <num_train_steps> <test_batch_size> 
```
- `{ | }`: Mutually exclusive items. Choose one from them.
- `< >`: Placeholder for which you must supply a value.
- `[ ]`: Optional items.

```
# WN18RR
bash runs.sh train ANStransE wn18rr 3 0 512 1024 300 6.0 0.5 0.0001 20000 8 --no_decay

# FB15k-237

bash runs.sh train ANStransE FB15k-237 2 0 1024 256 600 9.0 1.0 0.0001 1000 16

# YAGO3-10
bash runs.sh train ANStransE YAGO3-10 1 0 1024 256 900 24.0 1.0 0.0002 80000 4
```
