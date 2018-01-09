import numpy as np
import random
import pickle, glob
import scipy.io as pio


def l2_norm_512(input):


    for sample in range(0, input.shape[0]):
        for i in range(0,input.shape[1]):
            for j in range(0,input.shape[2]):
                norm = np.linalg.norm(input[sample,i,j,:])
                if norm==0:
                    continue
                else:
                    input[sample,i,j,:] = input[sample,i,j,:]/norm
    return input
def l2_norm_feat(input):


    for sample in range(0, input.shape[0]):

        norm = np.linalg.norm(input[sample,:])
        if norm==0:
            continue

        else:
            pass
        input[sample,:] = input[sample,:]/norm
    return input

def MatrixCompleter(input, wishingLen=10):
    leng = input.shape[0]
    output = np.zeros((wishingLen,4096))
    var = []
    if(wishingLen>leng):
        var = range(leng)
        while(len(var)<wishingLen):
            var = var + random.sample(range(leng),min(leng,(wishingLen-len(var))))
    else:
        var = range(leng)
        var = random.sample(var, wishingLen)
    var.sort()
    output = input[var,:]
    return output

def c3dseq_loader(vidnames,path = '/media/amir/2TF/mpii/c3d16_full/',wishingLen=10):
    output = np.zeros((len(vidnames),wishingLen,4096));
    for i in range(0, output.shape[0]):
        try:
            output[i,:,:] = MatrixCompleter(pio.loadmat(path+vidnames[i])['features'],wishingLen)

        except:

            print(vidnames[i])
            pass
    return output


def lebelsIndsToTrainInds(list_reduces, training_labels_indices):
    dicSize = len(list_reduces)
    training_labels = np.zeros((training_labels_indices.shape[0],dicSize))
    for s in range(0,training_labels_indices.shape[0]):
        t = training_labels_indices[s]
        if(t==-1):
            continue
        try:
            t2 = list_reduces.index(t)
            training_labels[s,t2] = 1
        except:
            pass
    return training_labels

def gen_val(main_path):
    list_reduces = pickle.load(open(main_path+'Answers_indices.pickle'))
    splits_path = main_path + 'test_patches/'
    print('in da validation generator')
    print(splits_path)
    files = glob.glob(splits_path+'test_labels_index_*')
    numberOfFiles = len(files)
    counter = 0
    step = 1
    while(True):
    #for i in range(0,3):


        i = counter%numberOfFiles


        left_sentences = np.load(splits_path+'left_indices_v2_'+str(i)+'.npy')
        right_sentences = np.load(splits_path+'right_indices_v2_'+str(i)+'.npy')
        training_labels_index = np.load(splits_path+'test_labels_index_'+str(i)+'.npy')
        training_labels = lebelsIndsToTrainInds(list_reduces, training_labels_index)

        visual_features = np.load(splits_path+'visualFeatures_512_'+str(i)+'.npy')
        c3d_seq = c3dseq_loader(np.load(splits_path+'vidnames_'+str(i)+'.npy').tolist())


        #visual_features = np.amax(visual_features,axis=1)
        #visual_features = np.amax(visual_features,axis=1)
        #c3d_seq = np.amax(c3d_seq,axis=1)


        yield ([left_sentences, right_sentences, visual_features,c3d_seq], [training_labels])
        counter = counter + 1



def gen(main_path):
    list_reduces = pickle.load(open(main_path+'Answers_indices.pickle'))
    splits_path = main_path + 'training_patches/'
    print(splits_path)
    splits_path_temporary = '/home/amir/Desktop/training_patches/'
    files = glob.glob(splits_path+'training_labels_index_*')
    print(splits_path+'training_labels_index_*')
    numberOfFiles = len(files)
    counter = 0
    step = 64
    khar = 0
    counter = 0
    choices = range(0, numberOfFiles)
    while(True):
    #for i in range(0,3):


        i = counter%numberOfFiles
        #i = random.choice(choices)
        i =  choices[i]
        left_sentences = np.load(splits_path+'left_indices_v2_'+str(i)+'.npy')
        right_sentences = np.load(splits_path+'right_indices_v2_'+str(i)+'.npy')
        training_labels_index = np.load(splits_path+'training_labels_index_'+str(i)+'.npy')
        training_labels = lebelsIndsToTrainInds(list_reduces, training_labels_index)


        visual_features = np.load(splits_path+'visualFeatures_512_'+str(i)+'.npy')
        c3d_seq = c3dseq_loader(np.load(splits_path+'vidnames_'+str(i)+'.npy').tolist())

        x =  range(0,left_sentences.shape[0])
        random.shuffle(x)

        left_sentences = left_sentences[x,:]
        right_sentences = right_sentences[x,:]
        visual_features = visual_features[x,:,:,:]
        c3d_seq = c3d_seq[x,:,:]
        training_labels = training_labels[x,:]



        #visual_features = np.amax(visual_features,axis=1)
        #visual_features = np.amax(visual_features,axis=1)
        #c3d_seq = np.amax(c3d_seq,axis=1)




        khar = 0
        while(khar< training_labels.shape[0]):
            m = min(khar+step,training_labels.shape[0])
            yield ([left_sentences[khar:m,:], right_sentences[khar:m,:],visual_features[khar:m,:,:,:], c3d_seq[khar:m,:,:]], [training_labels[khar:m,:]])
            #yield ([left_sentences[khar:m,:], right_sentences[khar:m,:],visual_features[khar:m,:], c3d_seq[khar:m,:]], [training_labels[khar:m,:]])
            khar = khar + step
        if(counter==0):
            direction = 1
            random.shuffle(choices)
        if(counter == (numberOfFiles-1)):
            direction = 0
            random.shuffle(choices)
        if(direction==1):
            counter = counter + 1
        if(direction==0):
            counter = counter - 1




def gen_val_positional(main_path):
    list_reduces = pickle.load(open(main_path+'Answers_indices.pickle'))
    splits_path = main_path + 'test_patches/'
    print('in da validation generator')
    print(splits_path)
    files = glob.glob(splits_path+'test_labels_index_*')
    numberOfFiles = len(files)
    counter = 0
    step = 1
    while(True):
    #for i in range(0,3):


        i = counter%numberOfFiles


        left_sentences = np.load(splits_path+'left_indices_v2_'+str(i)+'.npy')
        right_sentences = np.load(splits_path+'right_indices_v2_'+str(i)+'.npy')
        training_labels_index = np.load(splits_path+'test_labels_index_'+str(i)+'.npy')
        training_labels = lebelsIndsToTrainInds(list_reduces, training_labels_index)

        visual_features = np.load(splits_path+'visualFeatures_512_'+str(i)+'.npy')
        c3d_seq = c3dseq_loader(np.load(splits_path+'vidnames_'+str(i)+'.npy').tolist())


        #visual_features = np.amax(visual_features,axis=1)
        #visual_features = np.amax(visual_features,axis=1)
        #c3d_seq = np.amax(c3d_seq,axis=1)


        yield ([left_sentences, right_sentences, visual_features,c3d_seq,positionMask(left_sentences),positionMask(right_sentences)], [training_labels])
        counter = counter + 1



def gen_positional(main_path):
    list_reduces = pickle.load(open(main_path+'Answers_indices.pickle'))
    splits_path = main_path + 'training_patches/'
    print(splits_path)
    splits_path_temporary = '/home/amir/Desktop/training_patches/'
    files = glob.glob(splits_path+'training_labels_index_*')
    print(splits_path+'training_labels_index_*')
    numberOfFiles = len(files)
    counter = 0
    step = 64
    khar = 0
    counter = 0
    choices = range(0, numberOfFiles)
    while(True):
    #for i in range(0,3):


        i = counter%numberOfFiles
        #i = random.choice(choices)
        i =  choices[i]
        left_sentences = np.load(splits_path+'left_indices_v2_'+str(i)+'.npy')
        right_sentences = np.load(splits_path+'right_indices_v2_'+str(i)+'.npy')
        training_labels_index = np.load(splits_path+'training_labels_index_'+str(i)+'.npy')
        training_labels = lebelsIndsToTrainInds(list_reduces, training_labels_index)


        visual_features = np.load(splits_path+'visualFeatures_512_'+str(i)+'.npy')
        c3d_seq = c3dseq_loader(np.load(splits_path+'vidnames_'+str(i)+'.npy').tolist())

        x =  range(0,left_sentences.shape[0])
        random.shuffle(x)

        left_sentences = left_sentences[x,:]
        right_sentences = right_sentences[x,:]
        visual_features = visual_features[x,:,:,:]
        c3d_seq = c3d_seq[x,:,:]
        training_labels = training_labels[x,:]



        #visual_features = np.amax(visual_features,axis=1)
        #visual_features = np.amax(visual_features,axis=1)
        #c3d_seq = np.amax(c3d_seq,axis=1)




        khar = 0
        while(khar< training_labels.shape[0]):
            m = min(khar+step,training_labels.shape[0])
            yield ([left_sentences[khar:m,:], right_sentences[khar:m,:],visual_features[khar:m,:,:,:],\
                    c3d_seq[khar:m,:,:],positionMask(left_sentences[khar:m,:]),positionMask(right_sentences[khar:m,:])], [training_labels[khar:m,:]])
            #yield ([left_sentences[khar:m,:], right_sentences[khar:m,:],visual_features[khar:m,:], c3d_seq[khar:m,:]], [training_labels[khar:m,:]])
            khar = khar + step
        if(counter==0):
            direction = 1
            random.shuffle(choices)
        if(counter == (numberOfFiles-1)):
            direction = 0
            random.shuffle(choices)
        if(direction==1):
            counter = counter + 1
        if(direction==0):
            counter = counter - 1

def positionMask(myinput):
    output = np.zeros(myinput.shape)
    for i in range(0,myinput.shape[0]):
        counter = 1
        found = 0
        for j in range(0,myinput.shape[1]):
            if(myinput[i,j]==0):
                if(found):
                    output[i,j] = 31
                else:
                    output[i,j] = 0
            else:
                output[i,j] = counter
                counter = counter + 1
                found =1
    return output



if __name__ == "__main__":
     l = gen('/media/amir/2TF/mpii/')
     k = next(l)