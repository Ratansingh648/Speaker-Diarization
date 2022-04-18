# Speaker-Diarisation - "Who Spoke When?"
Speaker diarisation is the process of partitioning an input audio stream into homogeneous segments according to the speaker's identity. It can enhance the readability of an automatic speech transcription by structuring the audio stream into speaker turns and, when used together with speaker recognition systems, by providing the speaker’s true identity. It is used to answer the question "who spoke when?". 

Speaker diarisation is a combination of speaker segmentation and speaker clustering. The first aims at finding speaker change points in an audio stream. The second aims at grouping together speech segments on the basis of speaker characteristics. Speaker Diarisation finds its application in variety of tasks namely Multimedia information retrieval; speaker turn analysis and audio processing [1]. It improvises the performance of Automatic Speech transcription Systems when used with Speaker Recognition System [1,2]. Following image depicts the architecture followed by this code:


![image](https://user-images.githubusercontent.com/22644796/163759381-72f90030-9799-4ad3-98c1-eb8e1ca6c6b3.png)

We have taken a snippet from this <a href="https://www.youtube.com/watch?v=lhFU5H5KPFE" target="_blank"> Youtube Video</a> as an example. Author does not claim any rights to this video. Please enable audio while watching below example.

https://user-images.githubusercontent.com/22644796/163762535-660b27c5-dc00-4c71-948a-72674ead9b4b.mp4


# References
1. Q. Wang, C. Downey, L. Wan, P Mansfield, I Moreno, “Speaker Diarisation with LSTM ”
2. Zhu, Xuan; Barras, Claude; Meignier, Sylvain; Gauvain, Jean-Luc. "Improved speaker diarization using speaker identification". Retrieved 2012-01-25.
3. J. Fiscus “Fall 2004 Rich Transcription ( RT-04 F ) Evaluation Plan.”, National Institute of Standard and Technology
