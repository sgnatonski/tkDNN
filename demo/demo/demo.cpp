#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include <map>
#include <sstream>
#include <vector>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

#include "./messaging.cpp"

bool gRun;
bool SAVE_RESULT = false;
int frameCount = 0;
std::map<int, cv::Mat> frame_cache;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);

    std::string net = "yolo3_berkeley.rt";
    if(argc > 1)
        net = argv[1]; 
    std::string input = "../demo/yolo_test.mp4";
    if(argc > 2)
        input = argv[2]; 
    char ntype = 'y';
    if(argc > 3)
        ntype = argv[3][0]; 
    int n_classes = 80;
    if(argc > 4)
        n_classes = atoi(argv[4]); 
    int n_batch = 1;
    if(argc > 5)
        n_batch = atoi(argv[5]);
    float conf_thresh=0.3;
    if(argc > 6)
        conf_thresh = atof(argv[6]);
    bool show = true;
    if(argc > 7)
        show = atoi(argv[7]);     
    bool gstreamer = true;
    if(argc > 8)
        gstreamer = atoi(argv[8]); 
  
    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    //if(!show)
    //    SAVE_RESULT = true;
    
    connectNats(argc, argv);

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;  

    tk::dnn::DetectionNN *detNN;  

    switch(ntype)
    {
        case 'y':
            detNN = &yolo;
            break;
        case 'c':
            detNN = &cnet;
            break;
        case 'm':
            detNN = &mbnet;
            n_classes++;
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }

    detNN->init(net, n_classes, n_batch, conf_thresh);

    gRun = true;

    cv::VideoCapture cap;
    std::cout<<"setting camera buffer size = 1\n";
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    std::cout<<"opening camera\n";

    int numberOfFrames = 0;
    if (gstreamer){
        cap.open(input, cv::CAP_GSTREAMER);
    }        
    else {
        cap.open(input, cv::CAP_FFMPEG);
        numberOfFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    }

    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int f = cap.get(cv::CAP_PROP_FPS);
        
    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        resultVideo.open("cache.avi", cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(w, h));
    }

    cv::Mat frame;
    if(show){
        cv::namedWindow("detection", cv::WINDOW_NORMAL);
        cv::resizeWindow("detection", 800, 600);
    }

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    while(gRun) {
	    frameCount++;
        batch_dnn_input.clear();
        batch_frame.clear();
        //std::cout<<"reading frame"<<frameCount<<"\n";
                
        try {
            for(int bi=0; bi< n_batch; ++bi){
                cap >> frame; 
                if(frame.empty()) 
                    break;
                
                batch_frame.push_back(frame);

                // this will be resized to the net format
                batch_dnn_input.push_back(frame.clone());
            } 
            if(frame.empty()) 
                continue;

            frame_cache[frameCount] = frame;
            if (frame_cache.size() > (10 * 25)){
                frame_cache.erase(frame_cache.begin());
            }
        
            //inference
            detNN->update(batch_dnn_input, n_batch);

            publishDetections(detNN->batchDetected, detNN->classesNames, frameCount, w, h);

            if (!gstreamer){
                //std::cout<<"Frame "<<frameCount<<"/"<<numberOfFrames<<"\n"; 
                if (frameCount % numberOfFrames == 0) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                }
            }
            
            if(show){
                detNN->draw(batch_frame);

                for(int bi=0; bi< n_batch; ++bi){
                    cv::imshow("detection", batch_frame[bi]);
                    cv::waitKey(1);
                }
            }
            if(n_batch == 1 && SAVE_RESULT)
                resultVideo << frame;
        }
        catch(...){
            std::cout<<"exception, skipping frame\n"; 
        }
    }

    std::cout<<"detection end\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;   
    
    if(SAVE_RESULT) {
        resultVideo.release();
    }

    closeNats();
    
    if(!frame.data) {
        std::cout<<"No frame could be captured, terminating\n";
        exit(2);
    }

    return 0;
}
