325 def all_feature1(fpath):
326     try:
327          x, sr = librosa.load(fpath, sr=conf.sr, res_type="kaiser_fast")
328     except:
329          print("Fail to decode")
330          return None
331     x_trimmed, _ = librosa.effects.trim(x, top_db=30)
332     #amplitude
333     hop_length = 512
334     S = np.abs(librosa.stft(x_trimmed, n_fft=1024, hop_length=512))
335     #emotion
336     aweightPower=AweightPower_extract(x_trimmed, sr)#(1,31) 
337     power=librosa.feature.rmse(S=S)#(1,31)intensity 
338     zero= librosa.feature.zero_crossing_rate(x_trimmed, frame_length=1024, hop_length=512)#(1,31)
339     #breath
340     mfcc12=librosa.feature.mfcc(y=x_trimmed, sr=sr, n_mfcc=13)[12]
341     delta_mfcc=librosa.feature.delta(mfcc12,mode="nearest")#(1,31)
342     delta_energy=librosa.feature.delta(power,mode="nearest")#(1,31)
343     #timber
344     feat2 = librosa.feature.spectral_centroid(S=S)
345     feat3 = librosa.feature.spectral_bandwidth(S=S)
346     feat4 = librosa.feature.spectral_flatness(S=S)
347     feat5 = librosa.feature.spectral_contrast(S=S)
348     #amplitude
349     S = np.abs(librosa.stft(x_trimmed, n_fft=1024, hop_length=512))
350     #rhyme
351     oenv = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S))
352     tempo=librosa.feature.tempogram(onset_envelope=oenv, sr=sr,hop_length=hop_length)#(384,31)
353     #pitch
354     pitches = pitch_detect(S, sr, 300, 3000)
355     chroma=librosa.feature.chroma_cens(y=x_trimmed, sr=sr)
356     #pre-process
357     mel = librosa.feature.melspectrogram(S=S)
358     logmel = librosa.amplitude_to_db(S=mel)##!
359     #normalize
360     audio_features = np.array([aweightPower,power,zero,mfcc12,delta_mfcc,delta_energy,feat2,feat3,feat4,feat5,oenv,tempo,pitches,chroma,logmel]) #(1028,31)
361     feat= np.vstack(audio_features)
362     feat_mean = np.mean(feat, axis=0)
363     feat_std = np.std(feat, axis=0)
364     feat = (feat - feat_mean)/(feat_std+1e-10)
365     feat=feat.T#(,feature_dim)
366     print(fpath, feat.shape)
367     return feat
