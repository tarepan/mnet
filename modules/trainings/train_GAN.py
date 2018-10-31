# Generative Adversarial Networks [arxiv](https://arxiv.org/abs/1406.2661)
# adversarial process || minimax two-player game (simultaneously train two models)
# , a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere.


最終的にどうなるのか
pdataとpgが完全に一致する (誰が一致すると判定するんだ？)
Dがデータ割合50:50を知っているならD(G(z))=0.5に収束する
むしろそれは完全なDが完成した状態なのでは？
Dが微妙な状態でもGが完全なることはありうる。それを不完全なDはどうする？
Gは暗示的にpgをもつ
Dは確率を吐き出す。これ基本
