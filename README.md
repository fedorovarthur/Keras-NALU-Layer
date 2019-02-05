# Keras Add-on

Keras implementation of variety of the newest layers, losses, activations, etc. from the recent research papers.

By now there is the implementation of:
1. Neural Arithmetic Logic Unit and Neural Accumulator, https://arxiv.org/pdf/1808.00508.pdf;
2. Gaussian Error Linear Units, https://arxiv.org/pdf/1606.08415.pdf, GELU had been 
extended with reparametrization trick to have learnable mu and sigma;
3. Relational Loss, https://arxiv.org/pdf/1802.03145.pdf;
4. Swish activation function, https://arxiv.org/pdf/1710.05941.pdf, Swish had been added in two variants: with constant 
beta and parametrized with learnable beta.