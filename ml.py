from transformers import pipeline

def summarizer2(rawdoc):
    text = rawdoc
    to_tokenize = text

    # More precisely, today, we will be performing text summarization with a pretrained Transformer.
    # While the code that we will write is really simple and easy to follow, the technology behind the
    # easy interface is complex. In this section, we will therefore take a look at Transformers first,
    # which are state-of-the-art in Natural Language Processing.
    # This is followed by taking a closer look at the two variations of the Transformer that lie at the basis of the pretrained one that we will use,
    # being the BERT and the GPT model architectures.
    # Having understood these basics, we'll move on and look at the BART model, which is the model architecture that underpins the easy summarizer that we will be using today.
    # We will see that BART combines a bidirectional BERT-like encoder with a GPT-like decoder, allowing us to benefit from BERT bidirectionality while being able to generate text, which is not one of BERT's key benefits. Once we understand BART intuitively,
    # we're going to take a look at the pretrained BART model - because BART itself is only an architecture.
    # We will take a look at the CNN / Daily Mail dataset, which is what our model has been trained on.
    # Once we understand all these aspects, we can clearly see how our summarizer works, why it works, and then we can move to making it work. Let's go!


    # Initialize the HuggingFace summarization pipeline
    summarizer = pipeline("summarization")
    summarized = summarizer(to_tokenize, min_length=75, max_length=150)

    # Print summarized text
    doc = summarized[0]['summary_text'].replace(' .', '.\n')

    return doc, rawdoc, len(rawdoc.split(' ')), len(doc.split(' ')), len(rawdoc.split('.')), len(doc.split('.'))



# link https://github.com/christianversloot/machine-learning-articles/blob/main/easy-text-summarization-with-huggingface-transformers-and-machine-learning.md
