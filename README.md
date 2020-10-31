This is CS230, Deep-Neural-Vision project repo. we will be keep adding models, results, as we keep performing experiments here..
Currently we tried [_Show, Attend, and Tell_](https://arxiv.org/abs/1502.03044) paper architecture in PyTorch, with additional Beam search in it.

For Metrics we used pycocoeval & it's wrapper pip package.


This model learns _where_ to look.
As you generate a caption, word by word, you can see the model's gaze shifting across the image.
This is possible because of its _Attention_ mechanism, which allows it to focus on the part of the image most relevant to the word it is going to utter next.
Here are some captions generated on _test_ images not seen during training or validation
