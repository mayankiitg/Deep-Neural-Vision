This is CS230, Deep-Neural-Vision project repo. we will be keep adding models, results, as we keep performing experiments here..
Currently we tried [_Show, Attend, and Tell_](https://arxiv.org/abs/1502.03044) paper architecture in PyTorch, with additional Beam search in it.

For Metrics we used pycocoeval & it's wrapper pip package.


This model learns _where_ to look.
As you generate a caption, word by word, you can see the model's gaze shifting across the image.
This is possible because of its _Attention_ mechanism, which allows it to focus on the part of the image most relevant to the word it is going to utter next.
Here are some captions generated on _test_ images not seen during training or validation:

---

![](./img/plane.png)

---

![](./img/boats.png)

---

![](./img/bikefence.png)

---

![](./img/sheep.png)

---

![](./img/babycake.png)

---

![](./img/dogtie.png)

---


# Concepts

* **Image captioning**. duh.

* **Encoder-Decoder architecture**. Typically, a model that generates sequences will use an Encoder to encode the input into a fixed form and a Decoder to decode it, word by word, into a sequence.

* **Attention**. The use of Attention networks is widespread in deep learning, and with good reason. This is a way for a model to choose only those parts of the encoding that it thinks is relevant to the task at hand. The same mechanism you see employed here can be used in any model where the Encoder's output has multiple points in space or time. In image captioning, you consider some pixels more important than others. In sequence to sequence tasks like machine translation, you consider some words more important than others.

* **Transfer Learning**. This is when you borrow from an existing model by using parts of it in a new model. This is almost always better than training a new model from scratch (i.e., knowing nothing). As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand. Using pretrained word embeddings is a dumb but valid example. For our image captioning problem, we will use a pretrained Encoder, and then fine-tune it as needed.

* **Beam Search**. This is where you don't let your Decoder be lazy and simply choose the words with the _best_ score at each decode-step. Beam Search is useful for any language modeling problem because it finds the most optimal sequence.


### Encoder

The Encoder **encodes the input image with 3 color channels into a smaller image with "learned" channels**.

This smaller encoded image is a summary representation of all that's useful in the original image.

Since we want to encode images, we use Convolutional Neural Networks (CNNs).

We don't need to train an encoder from scratch. Why? Because there are already CNNs trained to represent images.

For years, people have been building models that are extraordinarily good at classifying an image into one of a thousand categories. It stands to reason that these models capture the essence of an image very well.

I have chosen to use the **101 layered Residual Network trained on the ImageNet classification task**, already available in PyTorch. As stated earlier, this is an example of Transfer Learning. You have the option of fine-tuning it to improve performance.

![ResNet Encoder](./img/encoder.png)

These models progressively create smaller and smaller representations of the original image, and each subsequent representation is more "learned", with a greater number of channels. The final encoding produced by our ResNet-101 encoder has a size of 14x14 with 2048 channels, i.e., a `2048, 14, 14` size tensor.

I encourage you to experiment with other pre-trained architectures. The paper uses a VGGnet, also pretrained on ImageNet, but without fine-tuning. Either way, modifications are necessary. Since the last layer or two of these models are linear layers coupled with softmax activation for classification, we strip them away.

### Decoder

The Decoder's job is to **look at the encoded image and generate a caption word by word**.

Since it's generating a sequence, it would need to be a Recurrent Neural Network (RNN). We will use an LSTM.

In a typical setting without Attention, you could simply average the encoded image across all pixels. You could then feed this, with or without a linear transformation, into the Decoder as its first hidden state and generate the caption. Each predicted word is used to generate the next word.

![Decoder without Attention](./img/decoder_no_att.png)

In a setting _with_ Attention, we want the Decoder to be able to **look at different parts of the image at different points in the sequence**. For example, while generating the word `football` in `a man holds a football`, the Decoder would know to focus on – you guessed it – the football!

![Decoding with Attention](./img/decoder_att.png)

Instead of the simple average, we use the _weighted_ average across all pixels, with the weights of the important pixels being greater. This weighted representation of the image can be concatenated with the previously generated word at each step to generate the next word.

### Attention

The Attention network **computes these weights**.

Intuitively, how would you estimate the importance of a certain part of an image? You would need to be aware of the sequence you have generated _so far_, so you can look at the image and decide what needs describing next. For example, after you mention `a man`, it is logical to declare that he is `holding a football`.

This is exactly what the Attention mechanism does – it considers the sequence generated thus far, and _attends_ to the part of the image that needs describing next.

![Attention](./img/att.png)

We will use _soft_ Attention, where the weights of the pixels add up to 1. If there are `P` pixels in our encoded image, then at each timestep `t` –

<p align="center">
<img src="./img/weights.png">
</p>

You could interpret this entire process as computing the **probability that a pixel is _the_ place to look to generate the next word**.

### Putting it all together

It might be clear by now what our combined network looks like.

![Putting it all together](./img/model.png)

- Once the Encoder generates the encoded image, we transform the encoding to create the initial hidden state `h` (and cell state `C`) for the LSTM Decoder.
- At each decode step,
  - the encoded image and the previous hidden state is used to generate weights for each pixel in the Attention network.
  - the previously generated word and the weighted average of the encoding are fed to the LSTM Decoder to generate the next word.

### Beam Search

We use a linear layer to transform the Decoder's output into a score for each word in the vocabulary.

The straightforward – and greedy – option would be to choose the word with the highest score and use it to predict the next word. But this is not optimal because the rest of the sequence hinges on that first word you choose. If that choice isn't the best, everything that follows is sub-optimal. And it's not just the first word – each word in the sequence has consequences for the ones that succeed it.

It might very well happen that if you'd chosen the _third_ best word at that first step, and the _second_ best word at the second step, and so on... _that_ would be the best sequence you could generate.

It would be best if we could somehow _not_ decide until we've finished decoding completely, and **choose the sequence that has the highest _overall_ score from a basket of candidate sequences**.

Beam Search does exactly this.

- At the first decode step, consider the top `k` candidates.
- Generate `k` second words for each of these `k` first words.
- Choose the top `k` [first word, second word] combinations considering additive scores.
- For each of these `k` second words, choose `k` third words, choose the top `k` [first word, second word, third word] combinations.
- Repeat at each decode step.
- After `k` sequences terminate, choose the sequence with the best overall score.

![Beam Search example](./img/beam_search.png)
