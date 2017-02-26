# faster-rcnn-cplusplus
faster-rcnn cplusplus in windows with visual studio 2013

the project is according to the http://blog.csdn.net/oYangZi12/article/details/53290426?locationNum=5&fps=1

#About the platform
You need a VC compiler to build these project, Visual Studio 2013 Community should be fine. You can 
download from https://www.visualstudio.com/downloads/.

#About Models
We choose the matlab model file and the prototxt file。 in matlab ,fter-rcnn  divided into two parts, 
rpn_net and faster-rcnn_net;thus there are two models and two prototxt file in matlab;these project is 
just a test project,if you want to train your own model , prehaps you need do it on matlab-faster-rcnn. 
and put the model in these project for test;

#About mean_images
 mean_images is converte from the model.mat in matlab . can save the mean_image by command imwrite(uint8(proposal_detection_model.image_means),'mean_image.bmp') or load the model.mat and save it.

#About Caffe 
i use the Microsoft version , https://github.com/Microsoft/caffe . if you want to use the faster-rcnn  you
need to add some others file to libcaffe project , in fact ,i add all of the head file in /include/caffe/
and all of the source file and cu file in /src/caffe/ to libcaffe project and rebuild it;
 
#About result
it's a little different from the matlab version .and it cost much time,maybe we can optimize the OpenCV parts
 
#Aboud 3rdparty
i have sorted out the thrid party depenendent library files in faster_3rdparty ,it is a release version , you 
can put  it in the caffe-master folder. and we should create a new porject to test faster-rcnn in 
caffe-master/windows/folder.As you can see, you can compile the project directly. What you need to do is only
change some necessary paths. All the code has been tested and the test results are also included. Wish it can 
help you .if it's helpful to you ,please give me a star thanks~


  
