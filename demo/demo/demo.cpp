#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include <map>

extern "C" {
#include <nats/nats.h>
}

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

bool gRun;
bool SAVE_RESULT = false;
int frameCount = 0;
std::map<int, cv::Mat> frame_cache;
int jpeg_quality = 70;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

static void
onMsg(natsConnection *nc, natsSubscription *sub, natsMsg *msg, void *closure)
{
    int f_no = atoi(natsMsg_GetData(msg));

    auto itr = frame_cache.find(f_no);

    if (itr == frame_cache.end()){
        std::cout<<"frame " << f_no << " not found in cache\n";

        natsConnection_Publish(nc, natsMsg_GetReply(msg), NULL, 0);
        natsMsg_Destroy(msg);
        return;
    }

    cv::Mat frame = itr->second;

    if (frame.empty()){
        std::cout<<"frame " << f_no << " found in cache but probably is now removed (too old)\n";

        natsConnection_Publish(nc, natsMsg_GetReply(msg), NULL, 0);
        natsMsg_Destroy(msg);
        return;
    }

    std::vector<uchar> buff;//buffer for coding
    std::vector<int> param(2);
    param[0] = cv::IMWRITE_JPEG_QUALITY;
    param[1] = jpeg_quality;//default(95) 0-100
    cv::imencode(".jpg", frame, buff, param);

    std::cout<<"replying with " << buff.size() << " bytes\n";

    natsConnection_Publish(nc, natsMsg_GetReply(msg), buff.data(), buff.size());
    natsMsg_Destroy(msg);
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
    std::string nats_url = NATS_DEFAULT_URL;
    if(argc > 3)
        nats_url = argv[3]; 
    char ntype = 'y';
    if(argc > 4)
        ntype = argv[4][0]; 
    int n_classes = 80;
    if(argc > 5)
        n_classes = atoi(argv[5]); 
    int n_batch = 1;
    if(argc > 6)
        n_batch = atoi(argv[6]); 
    bool show = true;
    if(argc > 7)
        show = atoi(argv[7]); 
    if(argc > 8)
        jpeg_quality = atoi(argv[8]); 
    
    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    //if(!show)
    //    SAVE_RESULT = true;
    
    natsConnection      *conn = NULL;
    natsSubscription    *sub  = NULL;
    natsMsg             *msg  = NULL;
    natsStatus          s;

    std::cout<<"connecting to NATS on " << nats_url << "\n";
    s = natsConnection_ConnectTo(&conn, nats_url.c_str());
    if (s != NATS_OK){
        nats_PrintLastErrorStack(stderr);
        exit(2);
    }

    s = natsConnection_Subscribe(&sub, conn, "frame", onMsg, NULL);
    if (s != NATS_OK){
        nats_PrintLastErrorStack(stderr);
        exit(2);
    }

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

    detNN->init(net, n_classes, n_batch);

    gRun = true;

    cv::VideoCapture cap;
    std::cout<<"setting camera buffer size = 1\n";
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    std::cout<<"opening camera\n";
    cap.open(input, cv::CAP_GSTREAMER);

    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int f = cap.get(cv::CAP_PROP_FPS);
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
            std::string dets = yolo.getDetections(frameCount);

            s = natsConnection_PublishString(conn, "detections", dets.c_str());
            //std::cout<<dets<<"\n";
            
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

    natsConnection_Destroy(conn);
    if (s != NATS_OK)
    {
        nats_PrintLastErrorStack(stderr);
        exit(2);
    }
    
    if(!frame.data) {
        std::cout<<"No frame could be captured, terminating\n";
        exit(2);
    }

    return 0;
}
