# Qualitative Analysis 

## What to log? 

### Category-1: Data Loading 

#### Category-1.1: Data reading/loading  

Logging needs to be enabled for **data reading** because **data reading** methods can facilitate **poisoning attacks** as observed by **[19] of Papernot-SoK-EuroS&P** and **speech2label and speech2text attack** as observed by **(Alzantot et al., 2018), (Cisse et al., 2017), Carlini & Wagner, 2018) of DawnSongAudioAttack.pdf**

Candidate code elements: 

> data.load(data.dset_dir), tmp_dict = pickle.load(f), checkpoint = torch.load(checkpoint_path), data = load_audio(args.input_path), load_randomly_augmented_audio(audio path, self.sample_rate), load_audio(audio_path), tarfile.open(target_file), get_dataset(), audio.load_wav(wav_path), with open("cmuarctic.data", "r") as f:, utils.pickle_load(filename), cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE), load_attribute_dataset(args.attr_file), datasets.ImageFolder(), torch.utils.data.DataLoader(), Image.open(args.demo_image), Image.fromarray(mask_copy.astype(np.uint8)), load_image(), pickle.load(f), testLoader = DataLoader(), read_h5file(os.path.join(os.getcwd(), 'train.h5')), load_lua(args.input_t7), pd.read_csv(), glob.iglob(), h5py.File(), codecs.open(), scipy.io.loadmat(), load_gt_roidb(), agent.replay_buffer.load(self.rbuf_filename), Image.fromArray(), tf.gfile.GFile(), dataset.ReadDatasetFile(), tf.io.gfile.glob(), yaml.load()

#### Category-1.2: Model loading 

Logging needs to be enabled for **model loading** because **model reading** methods can facilitate **poisoning attacks** as observed by **[19] of Papernot-SoK-EuroS&P**

Candidate code elements:

> model_dir_path = patch_path('models'), ref = CaffeFunction('VGG_ILSVRC_19_layers.caffemodel'),load_state_dict(), model_from_json(open()), network.load_net(), vgg.load_from_npy_file(), caffe_parser.read_caffemodel() , tf.train.Checkpoint(), get_network(args.network_name), tfhub.load(), scipy.misc.imresize()


### Category-2: Data  downloads 

Logging needs to be enabled for **data downloading** because **data download** methods can facilitate **attacks due to malformed input** as observed by **Section III-A of Xiao-S&P-Workshop** 

Candidate code elements: 

> wget.download('http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz'), urllib.request.urlopen(request), model_zoo.load_url(url), prepare_url_image(url), urllib.urlretrieve(), agent.load(misc.download_model())




### Category-3: Environment dynamics [Not sure if this will be kept as it requires physical environments]

Logging needs to be enabled for **environment dyanmics used in reinforcement learning** because **environment dyanmics** are susceptible to the **env-search-bb attack** as observed by **Section 4.3 of DawnSongDeeepRL Paper** 
More on implementation: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

Candidate code elements: we need to log environment dynamics: (i) state, (ii) reward, (iii) action, and (iv) policy = <observation, action, reward, done>

> *inner_next_obs, inner_reward, done = wrapped_env.step(action)*
> *obs, reward, done, tt = env.step(action)* 
> *state, reward, done, _ = env.step(action)*
> *obs, reward, done, info = env.step(action)* 
> *next_state, reward, done, _ = env.step(action)* 
> *batch_ob, reward, done, info = self.env.step(action)* 
> *state, reward, done, _ = env.step(action)*
> *o, r, done, info = self.env.step(action)* 
> *policy = torch.load(args.model_path)* 
> *env = gym.make(params.env_name)* and *num_inputs = env.observation_space.shape[0]* and *num_outputs = env.action_space.shape[0]*


### Category-4: Action Selection 

Logging needs to be enabled for **action taken in reinforcement learning** because **action selections** are susceptible to the **act-nn-wb attack** as observed by **Section 4.2 of DawnSongDeeepRL Paper** 

Candidate code elements: `action` used in env.step()

> *obs, reward, done, tt = env.step(action)* 

### Category-5: State observations 

Logging needs to be enabled for **observations in reinforcement learning** because **observations** are susceptible to the **multiple white and black box attacks** as observed by **Section 4.1 of DawnSongDeeepRL Paper** 

Candidate code elements: `obs` or `state` or `observations` obtained from in env.step(action)[0]

> *obs, reward, done, tt = env.step(action)* 



### Category-6: Classification decision of DNNs

Logging needs to be enabled for **classification decision provided by DNNs** because **classification decisions** are changed by the **DeepFool attack** as reported by **Section 2 of DeepFool CVPR paper** 

Candidate code elements:
-  `torch.nn` imported and used to perform classifications ... relatively harder to detect as many scripts are libraries, they provide utils but do not actually apply the library for classification decision 
- `from keras.models import *` Reference: https://realpython.com/python-keras-text-classification/ 


> *import torch.nn* or *import torch* and *nn.ReLU()* and *cls = PointNetCls(k = 5)* *out, _, _ = cls(sim_data)*
> *model=cascaded_model(D_m)* and *G=model(D, label_images)* and *Generator=G.permute(0,2,3,1)* and *output=np.minimum(np.maximum(Generator,0.0), 255.0)*
> *model = Model(inputs=[inputs], outputs=[conv10])* and *model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])* and *from keras.models import Model*
> *model = Graph()* and *from keras.models import *graph.compile()* and *graph.fit()* and *graph.predict()* 
> *model_from_file.fit()* and *model_from_file.evaluate()* and *from keras.models import Sequential, model_from_json*
> *graph = VGG_16_graph()* and *model = Graph()* and *from keras.models import ** and *model.compile()* and *graph.predict()* 




### Category-7: Label manipulation 

Logging needs to be enabled for **labels in datasets** because **labels** are manipulated by the **label perturbation attack** as reported by **Section 4.1 of Papernot SoK Survey Paper** 

Candidate code elements:
- variable names with `label(s)` 


> *train_data, train_label = read_h5file(os.path.join(os.getcwd(), 'train.h5'))*
> *val_data, val_label = read_h5file(os.path.join(os.getcwd(), 'val.h5'))*
> *label = np.array(hf.get('label'))* 
> *label = load_image(f).convert('P')* 
> *label = scipy.io.loadmat('{}/segmentation/img_{}.mat'.format(self.nyud_dir, idx))['segmentation'].astype(np.uint8)* 
> *label = os.path.basename(os.path.dirname(one_file))* 
> *raw_data,raw_label = load_data_and_labels(fenci_right_save_path,fenci_wrong_save_path)*
> *label = hfw.create_dataset("labels", data=df_attr[list_col_labels].values)*
> **



### Category-8: Incomplete Logging 

Logging attributes are missing: 
- (i) IP address logging needs to be enabled so that we can track *incoming IP addresses [who]* as advised by **App Logging S&P Paper** 
- (ii) timestamp logging needs to be enabled so that we can track *timestamp [when]* as advised by **App Logging S&P Paper** 

Candidate code elements:
- variable names with `import logging` 


> *logging.getLogger().setLevel(logging.DEBUG)*
> *logger = logging.getLogger() #no mention of timestamp*
> *logging.basicConfig() # timestamp not specified using FORMAT string*
> *from symnet.logger import logger* and *logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))* 
> *import tensorflow.compat.v1 as tf* and *tf.logging()* 
> *tf.compat.v1.logging.info("Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size))* and *import tensorflow as tf*
> 

A good example that should happen : 
> *FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'  logging.basicConfig(format=FORMAT)*




### Category-9: Detect absence of Logging for Exception Handling 

> A lot of prior work has observed that exception handling is a common target of 
logging, and absence of logging for exception is considered as a negative deevlopment practice. We will detect where exceptions are printed but not logged. Exceptions can happen for many reasons and not all of them are necessary to log. But the excpetions that developers already are capturing using `catch` need to be checked to see if they use logging for them. 

> For this you will check `except` blocks and see what is inside the `except` block. If there is no logging then report an antipattern. 