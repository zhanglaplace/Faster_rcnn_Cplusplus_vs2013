# Faster-Rcnn-cplusplus
  Faster-rcnn cplusplus in windows with visual studio 2013 the project is according to the http://blog.csdn.net/oYangZi12/article/details/53290426?locationNum=5&fps=1

  you can test with python model according https://github.com/zhanglaplace/faster-rcnn-cplusplus2
  
# Platform
  You need a VC compiler to build these project, Visual Studio 2013 Community should be fine. You can download from https://www.visualstudio.com/downloads/.

# Caffe 
  i use the Microsoft version , ![https://github.com/Microsoft/caffe](https://github.com/Microsoft/caffe) . if you want to use the faster-rcnn you need to add some others file to libcaffe project , in fact ,i add all of the head file in /include/caffe/
and all of the source file and cu file in /src/caffe/ to libcaffe project and rebuild it;Specifically,See: http://www.cnblogs.com/LaplaceAkuir/p/6445189.html
 
# Result
  it's a little different from the matlab version .and it cost much time. 
  
In My compute(GTX760 GPU) a picture of size(375\*500\*3)cost 246ms .
![image](https://github.com/zhanglaplace/Faster_rcnn_Cplusplus_vs2013/blob/master/imgs/result_004545.jpg)
![image](https://github.com/zhanglaplace/Faster_rcnn_Cplusplus_vs2013/blob/master/imgs/result_001150.jpg)
![image](https://github.com/zhanglaplace/Faster_rcnn_Cplusplus_vs2013/blob/master/imgs/result_000456.jpg)
![image](https://github.com/zhanglaplace/Faster_rcnn_Cplusplus_vs2013/blob/master/imgs/result_000542.jpg)
![image](https://github.com/zhanglaplace/Faster_rcnn_Cplusplus_vs2013/blob/master/imgs/result_001763.jpg)

# Mean_images
  mean_images is converte from the model.mat in matlab . you can save the mean_image by command imwrite(uint8(proposal_detection_model.image_means),'mean_image.bmp') 
 or load the model.mat and save it.
 
# Faster-RCNN Model
  We choose the matlab model file and the prototxt file. in matlab ,faster-rcnn  divided into two parts, 
rpn_net and faster-rcnn_net;thus there are two models and two prototxt files in matlab;these project is 
just a test demo,if you want to train your own model , prehaps you need do it on matlab-faster-rcnn. 
and put the model in these project,you can download the model:![http://pan.baidu.com/s/1dF88JvV
](http://pan.baidu.com/s/1dF88JvV)

# Caffe-3rdparty
  i have sorted out the thrid party depenendent library files in faster_3rdparty,![http://pan.baidu.com/s/1qYttnsS](http://pan.baidu.com/s/1qYttnsS)
password:*d0ud*,it is a release version , you can put it in the caffe-master folder. and we should create a new 
porject to test faster-rcnn in caffe-master/windows/folder.As you can see, you can compile the project 
directly. What you need to do is only change some necessary paths. All the code has been tested and the test 
results are also included.

# Something else
  if you want to test the project using your own model ,please modify the classNum to your class number and each class' name in  caffe_net.cpp.
finally,wish it can help you .if it's helpful to you ,please give me a star thanks~


  
