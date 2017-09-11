# NextLine Prediction

[TOC]

## Dataset

Considering non-existent or few online sources that provide rap lyrics, We obtain a corpus of rap lyrics. This corpus consists of 65,730 songs from 3,154 rappers from a lyrics website, [OHHLA](http://ohhla.com/inder.html).

Because of the size limit of file to be uploaded, we will upload the dataset to :

## Baseline Models

- **DopeSemantic**, which includes BOW, BOW5,and LSA;
- **DopeRhyme**, which includes EndRhyme, EndRhyme-1, and OtherRhyme; and
- **DopeLearning**, which is a combination of DopeSemantic, DopeRhyme, and LineLength.
- **DopeSemantic**, **DopeRhyme**, and **DopeLearning** are introduced in [ (Malmi et al. 2016)](https://github.com/ekQ/dopelearning), and called **Dopes** for short. 
- **Doc2vec**, which includes mere doc2vec representations;
- **Rhyme2vec**, which includes mere rhyme2vec representations;
- **con**, which includes $d_r+d_s$ dimensional representations, produced by concatenating doc2vec and rhyme2vec together.
- **conAE**, which employs an autoencoder(AE) network following the concatenating vectors; and
- **conVAE**, which uses a VAE network to replace the AE network. 

Note that, NN5 is not included in DopeLearning. Because the source code of NN5 is not available and we couldn't create a replication simply using the descriptions in its original paper. Furthermore, the increase produced by NN5 against DopeLearning of mean rank is $12.3$, the increase of MRR is $0.02$, as it reported. While the smallest increase of our methods against DopeLearning is more than $51$ and $0.06$ respectively. Beyond that, we have conduct another evaluation task in Section 4.4 of the paper, the baseline verses are provided by the original authors, using DopeLearning and NN5.

./rhyme2vec is experiment conducted on rhyme2vec, whose results are shown in table 2 of the paper.

## Run

For the main experiment just run the following command:
> python main.py

For the experiment in ./rhyme2vec:
> python rhyme2vec/main.py
