import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu
from keras.callbacks import EarlyStopping
from nltk.util import ngrams
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
import sys, time, os, warnings
import numpy as np
from keras.applications import VGG16
from keras import layers
from keras.layers import Dropout
import pandas as pd
from collections import Counter
from keras import models
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from nltk.translate.bleu_score import SmoothingFunction
from collections import OrderedDict
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import model_from_json
import sklearn
from sklearn.decomposition import PCA
import pickle
import string
from keras.preprocessing.text import Tokenizer
dir_Flickr_jpg = "/AD-HOME/sfarza3/Research/Python/project/Flicker8k_Dataset/"
dir_Flickr_res = "/AD-HOME/sfarza3/Research/Python/project/result/"
dir_Flickr_text = "/AD-HOME/sfarza3/Research/Python/project/Flickr8k_text"
dir_vgg16 = "/AD-HOME/sfarza3/Research/Python/project/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
topn = 20
binwidth = 3

jpgs = os.listdir(dir_Flickr_jpg)
vocab_size=4476
maxlen=30
#print(len(jpgs))
# given the test set, calculate the BLEU score of the newly generated captions
def calculate_BLEU(tokenizer,fnm_test, di_test, dt_test):
    nkeep = 5
    pred_good, pred_bad, bleus = [], [], []
    count = 0
    with open(dir_Flickr_res+'index_word.pickle', 'rb') as f:
        index_word = pickle.load(f)
    for jpgfnm, image_feature, tokenized_text in zip(fnm_test[473:], di_test[473:], dt_test[473:]):
        count += 1
        print(count)
        # if count % 200 == 0:
        #     print("  %4.2f is done.."%(100 * count / float(len(fnm_test))))

        caption_true = [index_word[i] for i in tokenized_text]
        caption_true = caption_true[1:-1]  ## remove startreg, and endreg
        ## captions
        caption = predict_caption(tokenizer,image_feature.reshape(1, len(image_feature)))
        #print(caption)
        if caption:
            caption = caption.split()
            caption = caption[1:-1]
            ## remove startreg, and endreg
            with open(dir_Flickr_res + 'captions.txt', 'a') as f:
                str1 = ' '.join(caption)
                f.write(str1+os.linesep)

            with open(dir_Flickr_res + 'captions_true.txt', 'a') as f:
                str2 = ' '.join(caption_true)
                f.write(str2+os.linesep)
            with open(dir_Flickr_res + 'jpgfnms.txt', 'a') as f:

                f.write(jpgfnm+os.linesep)
            cc = SmoothingFunction()
            bleu = sentence_bleu([caption_true], caption)
            bleus.append(bleu)
            print("bleu is %3.2f" %(bleu))
            with open(dir_Flickr_res + 'BLEUs.txt', 'a') as f:

                f.write(str(bleu)+os.linesep)
            if bleu > 0.7 and len(pred_good) < nkeep:
                pred_good.append((bleu, jpgfnm, caption_true, caption))
            elif bleu < 0.3 and len(pred_bad) < nkeep:
                pred_bad.append((bleu, jpgfnm, caption_true, caption))
    # print("Mean BLEU : %4.3f" % np.mean(bleus))
    # print("good")
    # print(pred_good)
    # print("bad")
    # print(pred_bad)
    # with open(dir_Flickr_res + 'bleu_result.txt', 'wb') as f:
    #    # f.write("mean BLEU score")
    #     f.write(np.mean(bleus))
    #     f.close()
    with open(dir_Flickr_res + 'pred_good.pickle', 'wb') as f:

        pickle.dump(pred_good, f)
    with open(dir_Flickr_res + 'pred_bad.pickle', 'wb') as f:
        pickle.dump(pred_bad, f)


# The LSTM model predicting the next token in sequence
def predict_caption(tokenizer,image):
    '''
    image.shape = (1,4462)
    '''
    # Model reconstruction from JSON file
    with open(dir_Flickr_res+'model_architecture.json', 'rb') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(dir_Flickr_res+'model_weights.h5')
    in_text = 'startseq'
    with open(dir_Flickr_res+'index_word.pickle', 'rb') as f:
        index_word = pickle.load(f)
    #print(index_word)
    #print(model.summary())
    for iword in range(maxlen):
         sequence = tokenizer.texts_to_sequences([in_text])[0]
         sequence = pad_sequences([sequence],maxlen)
         yhat = model.predict([image,sequence],verbose=0)
         yhat = np.argmax(yhat)
         newword = index_word[yhat]
         in_text += " " + newword
         if newword == "endseq":
             return (in_text)


# this functions create plot of 5 test image with its generated caption
def generate_caption(tokenizer,di_test,fnm_test):
    npic = 5
    npix = 224
    target_size = (npix, npix, 3)
    file=[]
    cap=[]
    count = 1
    fig = plt.figure(figsize=(10, 20))
    for jpgfnm, image_feature in zip(fnm_test[:npic], di_test[:npic]):
        ## images
        filename = dir_Flickr_jpg + '/' + jpgfnm
        file.append(filename)
        image_load = load_img(filename, target_size=target_size)
        ax = fig.add_subplot(npic, 2, count, xticks=[], yticks=[])
        ax.imshow(image_load)
        count += 1

        ## captions
        caption = predict_caption(tokenizer,image_feature.reshape(1, len(image_feature)))
        cap.append(caption)
        ax = fig.add_subplot(npic, 2, count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0, 0.5, caption, fontsize=20)
        count += 1
    print(file)
    print(cap)
    plt.show()

#plot the loss curve of training and validation period of the model
def plot_loss():
    plt.figure(figsize=(20, 3))
    
    # plt.legend()
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.show()

    plt.plot([5.423545996149505, 4.487228098772844, 4.047827592428033, 3.7324066742795594, 3.4683727688601156],
             label='loss')
    plt.plot([4.859445277393495, 4.536023587690088, 4.417816836790361, 4.39643198125976, 4.418985195920379],
             label='val_loss')
    # print('hello')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
## given the train set and validation set, creates the merge model
def create_model(Ximage_train,Xtext_train,ytext_train,Ximage_val,Xtext_val,ytext_val):
    dim_embedding = 64
    layer_size=128

    input_image = layers.Input(shape=(Ximage_train.shape[1],))
    fimage = layers.Dense(layer_size, activation='relu', name="ImageFeature")(input_image)
    ## sequence model
    input_txt = layers.Input(shape=(maxlen,))
    #the model does not use any pretrained embedding layer, rather learn from the input text
    ftxt = layers.Embedding(vocab_size, dim_embedding, mask_zero=True)(input_txt)
    ftxt = Dropout(0.5)(ftxt)
    ftxt = layers.LSTM(layer_size, name="CaptionFeature")(ftxt)
    ## combined model for decoder
    decoder = layers.add([ftxt, fimage])
    decoder = layers.Dense(layer_size, activation='relu')(decoder)
    output = layers.Dense(vocab_size, activation='softmax')(decoder)
    model = models.Model(inputs=[input_image, input_txt], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    #print(model.summary())
    start = time.time()
    #earlystopping=EarlyStopping()
    hist = model.fit([Ximage_train, Xtext_train], ytext_train,
                     epochs=10, verbose=2,
                     batch_size=128,

                     validation_data=([Ximage_val, Xtext_val], ytext_val))
    # hist = model.fit([Ximage_train, Xtext_train], ytext_train,
    #     #                  epochs=50, verbose=2,
    #     #                  batch_size=128,
    #     #                  callbacks=[earlystopping],
    #     #                  validation_data=([Ximage_val, Xtext_val], ytext_val))
    end = time.time()
    ##saves the model weight to a file for future use
    model.save_weights(dir_Flickr_res+'model_weights_1.h5')

    # Save the model architecture
    with open(dir_Flickr_res+'model_architecture_1.json', 'w') as f:
        f.write(model.to_json())
    #plot_loss(hist)
    with open(dir_Flickr_res + 'hist_1.pickle', 'wb') as f:
         pickle.dump(hist.history,f)
    # print(hist.history)
    print("TIME TOOK %3.2f MIN" %((end - start) / 60))
    print(Ximage_train.shape, Xtext_train.shape, ytext_train.shape)
## given the train set and validation set, process the text and image file to make final
## train and validation data for the model
def final_preprocessing(dtexts,dimages):
    N = len(dtexts)
    print("# captions/images = %d" %(N))

    assert(N==len(dimages))
    Xtext, Ximage, ytext = [],[],[]
    for text,image in zip(dtexts,dimages):

        for i in range(1,len(text)):
            in_text, out_text = text[:i], text[i]
            in_text = pad_sequences([in_text],maxlen=maxlen).flatten()
            out_text = to_categorical(out_text,num_classes = vocab_size)

            Xtext.append(in_text)
            Ximage.append(image)
            ytext.append(out_text)

    Xtext  = np.array(Xtext)
    Ximage = np.array(Ximage)
    ytext  = np.array(ytext)
    #print(" %d %d %d"%(Xtext.shape,Ximage.shape,ytext.shape))
    return(Xtext,Ximage,ytext)
## split the data set in train test and validation set (60, 20,20)
def split_test_val_train(dtexts):
    prop_test, prop_val = 0.2, 0.2

    N = len(dtexts)
    Ntest, Nval = int(N * prop_test), int(N * prop_val)
    maxlen = np.max([len(text) for text in dtexts])
    return(dtexts[:Ntest],
           dtexts[Ntest:Ntest+Nval],
           dtexts[Ntest+Nval:])
##Change character vector to integer vector using Tokenizer and call the calculate_BLEU
##method to find final result of generating caption on test set and evaluate it
def preprocess(df_txt0):

    dimages, keepindex = [], []
    nb_word = 8000
    tokenizer = Tokenizer(nb_words=nb_word)
    with open(dir_Flickr_res+'image_feature.pickle', 'rb') as handle:
        images=pickle.load(handle)
    df_txt0 = df_txt0.loc[df_txt0["index"].values == "0", :] ##take 1 caption
    for i, fnm in enumerate(df_txt0.filename):
        if fnm in images.keys():
            dimages.append(images[fnm])
            keepindex.append(i)

    fnames = df_txt0["filename"].iloc[keepindex].values
    dcaptions = df_txt0["new_caption"].iloc[keepindex].values
    dimages = np.array(dimages)
    tokenizer.fit_on_texts(dcaptions)
    index_word = dict([(index, word) for word, index in tokenizer.word_index.items()])
    print(index_word)
    vocab_size = len(tokenizer.word_index) + 1
    print("vocabulary size : %d" %(vocab_size))
    dtexts = tokenizer.texts_to_sequences(dcaptions)
    # print(dtexts[:5])
    with open(dir_Flickr_res+'index_word.pickle', 'wb') as f:

          pickle.dump(index_word, f)
    # with open(dir_Flickr_res + 'di_test.pickle', 'rb') as f:
    #     di_test = pickle.load(f)
    #
    #
    # with open(dir_Flickr_res + 'fnm_test.pickle', 'rb') as f:
    #     fnm_test = pickle.load(f)
    with open(dir_Flickr_res + 'dtexts.pickle', 'wb') as f:

        pickle.dump(dtexts, f)
    with open(dir_Flickr_res + 'fnames.pickle', 'wb') as f:
        pickle.dump(fnames, f)
    with open(dir_Flickr_res + 'dimages.pickle', 'wb') as f:
        pickle.dump(dimages, f)
    dt_test,  dt_val, dt_train   = split_test_val_train(dtexts)
    di_test,  di_val, di_train   = split_test_val_train(dimages)
    fnm_test,fnm_val, fnm_train  = split_test_val_train(fnames)
    with open(dir_Flickr_res + 'dt_test.pickle', 'wb') as f:

        pickle.dump(dt_test, f)
    with open(dir_Flickr_res + 'dt_val.pickle', 'wb') as f:
        pickle.dump(dt_val, f)
    with open(dir_Flickr_res + 'dt_train.pickle', 'wb') as f:
        pickle.dump(dt_train, f)
    with open(dir_Flickr_res + 'di_test.pickle', 'wb') as f:

        pickle.dump(di_test, f)
    with open(dir_Flickr_res + 'di_val.pickle', 'wb') as f:
        pickle.dump(di_val, f)
    with open(dir_Flickr_res + 'di_train.pickle', 'wb') as f:
        pickle.dump(di_train, f)
    with open(dir_Flickr_res + 'fnm_test.pickle', 'wb') as f:

        pickle.dump(fnm_test, f)
    with open(dir_Flickr_res + 'fnm_val.pickle', 'wb') as f:
        pickle.dump(fnm_val, f)
    with open(dir_Flickr_res + 'fnm_train.pickle', 'wb') as f:
        pickle.dump(fnm_train, f)
    Xtext_train, Ximage_train, ytext_train = final_preprocessing(dt_train,di_train)
    Xtext_val,   Ximage_val,   ytext_val   = final_preprocessing(dt_val,di_val)
    with open(dir_Flickr_res + 'Xtext_train.pickle', 'wb') as f:

        pickle.dump(Xtext_train, f)
    with open(dir_Flickr_res + 'Ximage_train.pickle', 'wb') as f:
        pickle.dump(Ximage_train, f)
    with open(dir_Flickr_res + 'ytext_train.pickle', 'wb') as f:
        pickle.dump(ytext_train, f)
    with open(dir_Flickr_res + 'Xtext_val.pickle', 'wb') as f:

        pickle.dump(Xtext_val, f)
    with open(dir_Flickr_res + 'Ximage_val.pickle', 'wb') as f:
        pickle.dump(Ximage_val, f)
    with open(dir_Flickr_res + 'ytext_val.pickle', 'wb') as f:
        pickle.dump(ytext_val, f)

    return tokenizer
##Do photo features make sense?
def plot_pca(y_pca,target_size):
    ## some selected pictures that are creating clusters
    picked_pic = OrderedDict()
    picked_pic["red"] = [1502, 5298, 2070, 7545, 1965]
    picked_pic["green"] = [746, 7627, 6640, 2733, 4997]
    picked_pic["magenta"] = [5314, 5879, 310, 5303, 3784]
    picked_pic["blue"] = [4644, 4209, 7326, 7290, 4394]
    picked_pic["yellow"] = [5895, 9, 27, 62, 123]
    picked_pic["purple"] = [5087]
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(y_pca[:, 0], y_pca[:, 1], c="white")

    for irow in range(y_pca.shape[0]):
        ax.annotate(irow, y_pca[irow, :], color="black", alpha=0.5)
    for color, irows in picked_pic.items():
         for irow in irows:
             ax.annotate(irow, y_pca[irow, :], color=color)
    ax.set_xlabel("pca embedding 1", fontsize=30)
    ax.set_ylabel("pca embedding 2", fontsize=30)
    plt.show()

    ## plot of images
    fig = plt.figure(figsize=(16, 20))
    count = 1
    for color, irows in picked_pic.items():
        for ivec in irows:
            name = jpgs[ivec]
            filename = dir_Flickr_jpg + '/' + name
            image = load_img(filename, target_size=target_size)

            ax = fig.add_subplot(len(picked_pic), 5, count,
                                 xticks=[], yticks=[])
            count += 1
            plt.imshow(image)
            plt.title("{} ({})".format(ivec, color))
    plt.show()
## extract image features using VGG16 model and saved the feature valus in 'image_feature.pickle'
def image_feature():
    images = OrderedDict()
    npix = 224
    target_size = (npix, npix, 3)
    features=[]
    # data = np.zeros((len(jpgs), npix, npix, 3))
    # for i, name in enumerate(jpgs):
    #     # load an image from file
    #     filename = dir_Flickr_jpg + '/' + name
    #     image = load_img(filename, target_size=target_size)
    #     # convert the image pixels to a numpy array
    #     image = img_to_array(image)
    #     nimage = preprocess_input(image)
    #
    #     y_pred = modelvgg.predict(nimage.reshape((1,) + nimage.shape[:3]))
    #     images[name] = y_pred.flatten()
    # #encoder = np.array(images.values())
    # encoder = list(images.values())
    with open(dir_Flickr_res+'image_feature.pickle', 'rb') as handle:
        encoder=pickle.load(handle)

        # pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for detail in encoder.keys():
        #features.append(detail)
        print(detail)



    #print(encoder)
    # pca = PCA(n_components=2)
    # y_pca = pca.fit_transform(features)
    # plot_pca(y_pca, target_size)

##preprocess the image caption, clean ans add start and end symbol
def text_clean(text_original):
    text_no_punctuation = text_original.translate(str.maketrans('', '', string.punctuation))
    text_len_more_than1 = ""
    for word in text_no_punctuation.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    clean_text = ""
    for word in text_len_more_than1.split():
        isalpha = word.isalpha()

        if isalpha:
            clean_text += " " + word
    clean_text = 'startseq ' + clean_text + ' endseq'
    return (clean_text)
def data_preprocess():
    clean_txt=[]

    df_txt = create_dataframe()
    for i, caption in enumerate(df_txt.caption.values):
        newcaption = text_clean(caption)

        clean_txt.append(newcaption)
    df_txt["new_caption"] = clean_txt

    with open(dir_Flickr_res + 'dataframe.pickle', 'wb') as f:
        pickle.dump(df_txt, f)
    return df_txt
##plot the word frequency graph
def plthist(dfsub, title):
    plt.figure(figsize=(20, 3))
    plt.bar(dfsub.index, dfsub["count"])
    plt.yticks(fontsize=20)
    plt.xticks(dfsub.index, dfsub["word"], rotation=90, fontsize=20)
    plt.title(title, fontsize=20)
    plt.show()

##plot the caption length frequency
def plthist_length(df_txt, title):
    plt.figure(figsize=(10, 3))
    plt.hist(df_txt, bins=range(min(df_txt), max(df_txt) + binwidth, binwidth))
    plt.yticks(fontsize=15)
    plt.xlabel('caption length')
    plt.ylabel('count')

    plt.title(title, fontsize=20)
    plt.show()

##calculate word frequency vector of the captions
def df_word(df_txt):
    vocabulary = []
    for txt in df_txt.caption.values:
        vocabulary.extend(txt.split())
    print('Vocabulary Size: %d' % len(set(vocabulary)))
    ct = Counter(vocabulary)
    dfword = pd.DataFrame({"word": list(ct.keys()), "count": list(ct.values())})
    dfword = dfword.sort_values('count', ascending=False)
    dfword = dfword.reset_index()[["word", "count"]]
    return (dfword)
##create dataframe with the dataset information
def create_dataframe():
    file = open(dir_Flickr_text, 'r')

    text = file.read()
    file.close()

    datatxt = []
    for line in text.split('\n'):
        col = line.split('\t')
        if len(col) == 1:
            continue
        w = col[0].split("#")
        datatxt.append({'filename': w[0], 'index': w[1], "caption": col[1].lower(), "cap_len": len(col[1].split())})

    df_txt = pd.DataFrame(datatxt)
    return df_txt
def image_process():
    modelvgg = VGG16(include_top=True, weights=None)

    # ## load the locally saved weights
    modelvgg.load_weights(dir_vgg16)
    modelvgg.layers.pop()
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    modelvgg.summary()
    image_feature()
def call_model():
    with open(dir_Flickr_res + 'Ximage_train.pickle', 'rb') as f:
        Ximage_train = pickle.load(f)
    with open(dir_Flickr_res + 'Xtext_train.pickle', 'rb') as f:
        Xtext_train = pickle.load(f)
    with open(dir_Flickr_res + 'ytext_train.pickle', 'rb') as f:
        ytext_train = pickle.load(f)
    with open(dir_Flickr_res + 'Ximage_val.pickle', 'rb') as f:
        Ximage_val = pickle.load(f)
    with open(dir_Flickr_res + 'Xtext_val.pickle', 'rb') as f:
        Xtext_val = pickle.load(f)
    with open(dir_Flickr_res + 'ytext_val.pickle', 'rb') as f:
        ytext_val = pickle.load(f)
        create_model(Ximage_train, Xtext_train, ytext_train, Ximage_val, Xtext_val, ytext_val)
def model_test(tokenizer):


    with open(dir_Flickr_res + 'fnm_test.pickle', 'rb') as f:

         fnm_test = pickle.load(f)


    with open(dir_Flickr_res + 'di_test.pickle', 'rb') as f:
         di_test = pickle.load(f)

    with open(dir_Flickr_res + 'dt_test.pickle', 'rb') as f:
         dt_test = pickle.load(f)

         calculate_BLEU(tokenizer,fnm_test, di_test, dt_test)




def main():
    #create dataframe of image_file, caption
    df_txt=data_preprocess()
    #visualise word distribution in the image captions
    dfword = df_word(df_txt)
    plthist(dfword.iloc[:topn, :],
            title="The top 50 most frequently appearing words")
    plthist(dfword.iloc[-topn:, :],
            title="The least 50 most frequently appearing words")
    #load model VGG16 and compute image features
    image_process()
    #map image file with caption, split train, validation and test data
    tokenizer=preprocess(df_txt)
    #train final model, experiment with 3 diffrent models
    call_model()
    #generate caption for test images and calculate BLEU-4 score
    model_test(tokenizer)



if __name__=="__main__":
    main()