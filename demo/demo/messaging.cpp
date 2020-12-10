#include <sstream>
#include <vector>

extern "C" {
#include <nats/nats.h>
}

natsConnection      *conn = NULL;
natsSubscription    *sub  = NULL;
natsMsg             *msg  = NULL;
natsStatus          s;

extern std::map<int, cv::Mat> frame_cache;
int jpeg_quality = 70;

std::vector<std::string> split (const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

static void
onMsg(natsConnection *nc, natsSubscription *sub, natsMsg *msg, void *closure)
{
    auto payload = split(natsMsg_GetData(msg), ',');
    int f_no = atoi(payload[0].c_str());
    int w = payload.size() > 2 ? atoi(payload[1].c_str()) : 0;
    int h = payload.size() > 2 ? atoi(payload[2].c_str()) : 0;

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
    if (w > 0 && frame.size().width > w)
    {
        cv::Mat resized;
        double scale = float(w)/frame.size().width;
        cv::resize(frame, resized, cv::Size(0, 0), scale, scale);
        cv::imencode(".jpg", resized, buff, param);
    }
    else 
    {
        cv::imencode(".jpg", frame, buff, param);
    }

    std::cout<< f_no << ": replying with " << buff.size() << " bytes\n";

    natsConnection_Publish(nc, natsMsg_GetReply(msg), buff.data(), buff.size());
    natsMsg_Destroy(msg);
}

static std::string
getDetections(std::vector<std::vector<tk::dnn::box>> batchDetected, std::vector<std::string> classesNames, int frameNumber, int width, int height){
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    std::string det_class;
    float prob = 0;
    std::ostringstream result;
        
    result << "{";
    result << "\"fn\": " << frameNumber <<  "," << "\"width\": " << width <<  ","<< "\"height\": " << height <<  ",";
    result << "\"det\": [";
    for(int bi=0; bi<batchDetected.size(); ++bi){
        result << "[";
        for(int i=0; i<batchDetected[bi].size(); i++) {
            b           = batchDetected[bi][i];
            x0   		= b.x;
            x1   		= b.x + b.w;
            y0   		= b.y;
            y1   		= b.y + b.h;
            prob        = b.prob;
            det_class 	= classesNames[b.cl];
            result << "{ \"cl\": \"" << det_class << "\", " << "\"x0\": " << x0 << ", " << "\"x1\": " << x1 << ", " << "\"y0\": " << y0 << ", " << "\"y1\": " << y1 << ", " << "\"pr\": " << prob <<  " }";
            if (i + 1 < batchDetected[bi].size()){
                result << ",";
            }
        }
        result << "]";
        if (bi + 1 < batchDetected.size()){
            result << ",";
        }
    }
    result << "]";
    result << "}";
    return result.str();
}

static void publishDetections(std::vector<std::vector<tk::dnn::box>> batchDetected, std::vector<std::string> classesNames, int frameNumber, int width, int height){
    std::string dets = getDetections(batchDetected, classesNames, frameNumber, width, height);
    s = natsConnection_PublishString(conn, "detections", dets.c_str());
}

static void connectNats(int argc, char *argv[]){
    std::string nats_url = NATS_DEFAULT_URL;
    if(argc > 9)
        jpeg_quality = atoi(argv[9]);        
    if(argc > 10)
        nats_url = argv[10];    

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
}

static void closeNats(){
    natsConnection_Destroy(conn);
    if (s != NATS_OK)
    {
        nats_PrintLastErrorStack(stderr);
        exit(2);
    }
}