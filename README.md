# seq2seq


## RNN-Attention
```
# calcuate correlation with x,y inner product 
att = softmax(<encoded_ys,encoded_xs>)
xs_att = att * xs
output = [xs_att,ys] # concat
```