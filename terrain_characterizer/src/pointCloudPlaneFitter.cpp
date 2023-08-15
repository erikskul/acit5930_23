// Adapted from https://github.com/apalomer/plane_fitter

#include <iostream>
#include <fstream>

// ROS
#include <ros/ros.h>
#include <ros/publisher.h>
#include <sensor_msgs/PointCloud2.h>
#include <dynamic_reconfigure/server.h>
#include <terrain_characterizer/ErrorNav.h>

// PCL
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <std_msgs/Float64MultiArray.h>
#include <pcl/filters/crop_box.h>
#include "terrain_characterizer/algorithmParametersConfig.h"
// #include "terrain_characterizer/featureLoggingService.h"

double point2planedistance(pcl::PointXYZ pt, pcl::ModelCoefficients::Ptr coefficients){
    double f1 = fabs(coefficients->values[0]*pt.x+coefficients->values[1]*pt.y+coefficients->values[2]*pt.z+coefficients->values[3]);
    double f2 = sqrt(pow(coefficients->values[0],2)+pow(coefficients->values[1],2)+pow(coefficients->values[2],2));
    return f1/f2;
}

class ColorMap{
public:
    ColorMap(double mn, double mx): mn(mn), mx(mx){}
    void setMinMax(double min, double max){ mn = min; mx = max;}
    void setMin(double min){mn = min;}
    void setMax(double max){mx = max;}
    void getColor(double c,uint8_t& R, uint8_t& G, uint8_t& B){
        double normalized = (c - mn)/(mx-mn) * 2 - 1;
        R = (int) (base(normalized - 0.5) * 255);
        G = (int) (base(normalized) * 255);
        B = (int) (base(normalized + 0.5) * 255);
    }
    void getColor(double c, double &rd, double &gd, double &bd){
        uint8_t r;
        uint8_t g;
        uint8_t b;
        getColor(c,r,g,b);
        rd = (double)r/255;
        gd = (double)g/255;
        bd = (double)b/255;
    }
    uint32_t getColor(double c){
        uint8_t r;
        uint8_t g;
        uint8_t b;
        getColor(c,r,g,b);
        return ((uint32_t)r<<16|(uint32_t)g<<8|(uint32_t)b);
    }


private:
    double interpolate(double val, double y0, double x0, double y1, double x1){
        return (val - x0)*(y1-y0)/(x1-x0) + y0;
    }
    double base(double val){
        if (val <= -0.75) return 0;
        else if (val <= -0.25) return interpolate(val,0,-0.75,1,-0.25);
        else if (val <= 0.25) return 1;
        else if (val <= 0.75) return interpolate(val,1.0,0.25,0.0,0.75);
        else return 0;
    }
private:
    double mn,mx;
};

class Color{
private:
    uint8_t r;
    uint8_t g;
    uint8_t b;

public:
    Color(uint8_t R,uint8_t G,uint8_t B):r(R),g(G),b(B){

    }

    void getColor(uint8_t &R,uint8_t &G,uint8_t &B){
        R = r;
        G = g;
        B = b;
    }
    void getColor(double &rd, double &gd, double &bd){
        rd = (double)r/255;
        gd = (double)g/255;
        bd = (double)b/255;
    }
    uint32_t getColor(){
        return ((uint32_t)r<<16|(uint32_t)g<<8|(uint32_t)b);
    }
};

class pointCloudPlaneFitter{
public:
    pointCloudPlaneFitter(ros::NodeHandle nh): _nh(nh){
        initialize();
    }

    void initialize(){

        // Get node name
        _name = ros::this_node::getName();

        // Publishers
        _pub_inliers = _nh.advertise<sensor_msgs::PointCloud2 >("inliers",2);
        _pub_feature = _nh.advertise<terrain_characterizer::ErrorNav>("/terrain_nav/errors",1);
        _pub_croppedL = _nh.advertise<sensor_msgs::PointCloud2>("croppedL",2);
        _pub_croppedM = _nh.advertise<sensor_msgs::PointCloud2>("croppedM",2);
        _pub_croppedR = _nh.advertise<sensor_msgs::PointCloud2>("croppedR",2);
        _pub_outliers = _nh.advertise<sensor_msgs::PointCloud2>("outliers",2);

        // Subscriber
        _subs = _nh.subscribe("/camera/depth/color/points",1,&pointCloudPlaneFitter::pointCloudCb,this);

        //_max_distance = 0.035; // 20mm

        // Get parameters
        ros::param::param<bool>("~color_pc_with_error",_color_pc_with_error,false);
        ros::param::param<bool>("~enable_sending",_enable_sending,false);
        ros::param::param<bool>("~enable_crop",_enable_crop,false);

        ros::param::param<double>("~total_width",total_width,false);
        ros::param::param<double>("~max_distance",_max_distance,false);

        ros::param::param<double>("~minY_L",minY_L,false);
        ros::param::param<double>("~maxY_L",maxY_L,false);
        ros::param::param<double>("~minY_M",minY_M,false);
        ros::param::param<double>("~maxY_M",maxY_M,false);
        ros::param::param<double>("~minY_R",minY_R,false);
        ros::param::param<double>("~maxY_R",maxY_R,false);
        ros::param::param<double>("~minZ",minZ,false);
        ros::param::param<double>("~maxZ",maxZ,false);

        // Create dynamic reconfigure
        drCallback = boost::bind( &pointCloudPlaneFitter::updateParameters, this, _1, _2);
        drServer.setCallback(drCallback);

        // Create colors palette
        createColors();

        // Inform initialized
        ROS_INFO("%s: node initialized.",_name.c_str());
    }

    void updateParameters(terrain_characterizer::algorithmParametersConfig& config, uint32_t level){
        _color_pc_with_error = config.paint_with_error;
        _enable_sending = config.enable_sending;
        _max_distance = config.max_distance;
        total_width = config.total_width;
        minY_L = config.minY_L;
        maxY_L = config.maxY_L;
        minY_M = config.minY_M;
        maxY_M = config.maxY_M;
        minY_R = config.minY_R;
        maxY_R = config.maxY_R;
        minZ = config.minZ;
        maxZ = config.maxZ;
        
        // _enable_crop = config.enable_crop;
    }

    double getPercentile(std::vector<double> vector, float percentile){
        size_t size = vector.size();

        sort(vector.begin(), vector.end());

        return vector[int(size * percentile/100.0)];
    }

    void pointCloudCb(const sensor_msgs::PointCloud2::ConstPtr &msg){

        // Convert to pcl point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_msg (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg,*cloud_msg);
        ROS_DEBUG("%s: new pointcloud (%i,%i)(%zu)",_name.c_str(),cloud_msg->width,cloud_msg->height,cloud_msg->size());

        // Filter cloud
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_msg);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits(0.001,10000);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pass.filter (*cloud);

        // Get segmentation ready
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::ModelCoefficients::Ptr coefficientsL(new pcl::ModelCoefficients);
        pcl::ModelCoefficients::Ptr coefficientsM(new pcl::ModelCoefficients);
        pcl::ModelCoefficients::Ptr coefficientsR(new pcl::ModelCoefficients);
        std::vector<pcl::ModelCoefficients::Ptr> all_coefficients = {coefficientsL, coefficientsM, coefficientsR};
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::PointIndices::Ptr inliersL(new pcl::PointIndices);
        pcl::PointIndices::Ptr inliersM(new pcl::PointIndices);
        pcl::PointIndices::Ptr inliersR(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::ExtractIndices<pcl::PointXYZ> extract_out (true);
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold(_max_distance);

        // Create pointcloud to publish inliers
        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pub(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pub_L(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pub_M(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pub_R(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pub_out(new pcl::PointCloud<pcl::PointXYZRGB>);
        int original_size(cloud->height*cloud->width);

        // Fit the plane
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        // Crop original cloud into 3 segments
        double w1 = total_width / 2;
        double w2 = total_width / 6;

        std::vector<pcl::PointCloud<pcl::PointXYZ>> cropped_clouds;
        std::vector<pcl::PointIndicesPtr> cropped_inliers; 
        pcl::PointCloud<pcl::PointXYZ>::Ptr croppedL (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr croppedM (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr croppedR (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::CropBox<pcl::PointXYZ> boxFilter;
        // Left
        boxFilter.setMin(Eigen::Vector4f(-w1, minY_L, minZ, 1.0));
        boxFilter.setMax(Eigen::Vector4f(-w2, maxY_L, maxZ, 1.0));
        boxFilter.setInputCloud(cloud);
        boxFilter.filter(*croppedL);
        cropped_clouds.push_back(*croppedL);
        // Middle
        boxFilter.setMin(Eigen::Vector4f(-w2, minY_M, minZ, 1.0));
        boxFilter.setMax(Eigen::Vector4f(w2, maxY_M, maxZ, 1.0));
        boxFilter.setInputCloud(cloud);
        boxFilter.filter(*croppedM);
        cropped_clouds.push_back(*croppedM);
        // Right
        boxFilter.setMin(Eigen::Vector4f(w2, minY_R, minZ, 1.0));
        boxFilter.setMax(Eigen::Vector4f(w1, maxY_R, maxZ, 1.0));
        boxFilter.setInputCloud(cloud);
        boxFilter.filter(*croppedR);
        cropped_clouds.push_back(*croppedR);

        // Fit plane for each segment
        // Left
        seg.setInputCloud(croppedL);
        seg.segment(*inliersL, *all_coefficients[0]);
        cropped_inliers.push_back(inliersL);
        // Middle
        seg.setInputCloud(croppedM);
        seg.segment(*inliersM, *all_coefficients[1]);
        cropped_inliers.push_back(inliersM);
        // Right
        seg.setInputCloud(croppedR);
        seg.segment(*inliersR, *all_coefficients[2]);
        cropped_inliers.push_back(inliersR);

        // Initialize all error variables and vectors
        double mean_error_L(0), mean_error_M(0), mean_error_R(0);
        double MSE_L(0), MSE_M(0), MSE_R(0);
        double max_error_L(0), max_error_M(0), max_error_R(0);
        double min_error_L(100000), min_error_M(100000), min_error_R(100000);
        double sigma_L(0), sigma_M(0), sigma_R(0);
        std::vector<double> err_L, err_M, err_R;
        std::vector<double> mean_errors = {mean_error_L, mean_error_M, mean_error_R};
        std::vector<double> MSEs = {MSE_L, MSE_M, MSE_R};
        std::vector<double> max_errors = {max_error_L, max_error_M, max_error_R};
        std::vector<double> min_errors = {min_error_L, min_error_M, min_error_R};
        std::vector<double> sigmas = {sigma_L, sigma_M, sigma_R};
        std::vector<std::vector<double>> errs = {err_L, err_M, err_R};
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pub_clouds {cloud_pub_L, cloud_pub_M, cloud_pub_R};

        terrain_characterizer::ErrorNav featureMsg;

        // Iterate inliers to get error
        for (int j=0;j<cropped_inliers.size();j++) { // For each of three segments (left, middle, right)
            for (int i=0;i<cropped_inliers[j]->indices.size();i++){ // For each point in current inlier segment j

                // Get Point
                pcl::PointXYZ pt = cropped_clouds[j].points[cropped_inliers[j]->indices[i]];

                // Compute distance
                double d = point2planedistance(pt,all_coefficients[j])*1000;// mm
                errs[j].push_back(d);

                // Update statistics
                mean_errors[j] += d;
                MSEs[j] += pow(d, 2);
                if (d>max_errors[j]) max_errors[j] = d;
                if (d<min_errors[j]) min_errors[j] = d;
                

            }

            // Compute errors
            mean_errors[j] /= cropped_inliers[j]->indices.size();
            MSEs[j] /= cropped_inliers[j]->indices.size();
        }


        // Create a common ColorMap across the three segments
        double color_min = 1000.0;
        double color_max = 0;

        for (int j=0;j<cropped_inliers.size();j++) {
            if (min_errors[j] < color_min) color_min = min_errors[j];
            if (max_errors[j] > color_max) color_max = max_errors[j];
        }

        ColorMap cm(color_min,color_max);

        // Color all three segment pointclouds, and calculate final standard deviation
        for (int j=0;j<cropped_inliers.size();j++) { // For each of three segments (left, middle, right)
            for (int i=0;i<cropped_inliers[j]->indices.size();i++){ // For each point in current inlier segment j
                
                sigmas[j] += pow(errs[j][i] - mean_errors[j],2);
                
                if (_enable_sending) {
                    // Get Point
                    pcl::PointXYZ pt = cropped_clouds[j].points[cropped_inliers[j]->indices[i]];

                    // Copy point to new cloud
                    pcl::PointXYZRGB pt_color;
                    pt_color.x = pt.x;
                    pt_color.y = pt.y;
                    pt_color.z = pt.z;
                    uint32_t rgb;
                    if (_color_pc_with_error)
                        rgb = cm.getColor(errs[j][i]);
                    else
                        rgb = colors[0].getColor();
                    pt_color.rgb = *reinterpret_cast<float *>(&rgb);
                    pub_clouds[j]->points.push_back(pt_color);

                    
                }
            }
            
            sigmas[j] = sqrt(sigmas[j]/cropped_inliers[j]->indices.size());
            ROS_INFO("%d :%lu points, average %.2f, %.2f(mm), mse: %.2f, sd: %.2f (mm), %.1f%% of points",j,cropped_inliers[j]->indices.size(),double(mean_errors[j])/double(cropped_inliers[j]->indices.size()),mean_errors[j],MSEs[j],sigmas[j],(double(cropped_inliers[j]->indices.size()) / double(original_size))*100.0);
        }

        if (_enable_sending) {
            // Extract outliers, save as the new cloud. Before this, cloud saves the entire pointcloud, after this, it only has outlier points
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            pcl::PointCloud <pcl::PointXYZ> cloudF;
            extract.filter(cloudF);
            cloud->swap(cloudF);
        }

        // Display information
        /*ROS_INFO("%s: fitted plane: %fx%s%fy%s%fz%s%f=0 (inliers: %zu/%i)",
                 _name.c_str(),
                 coefficients->values[0],(coefficients->values[1]>=0?"+":""),
                 coefficients->values[1],(coefficients->values[2]>=0?"+":""),
                 coefficients->values[2],(coefficients->values[3]>=0?"+":""),
                 coefficients->values[3],
                 inliers->indices.size(),original_size);*/

        // ROS_INFO("%s: me: %lu points, %.2f(mm), mse: %.2f, sd: %.2f (mm), %.1f%% of points",_name.c_str(),inliers->indices.size(),mean_error,MSE,sigma,(double(inliers->indices.size()) / double(original_size))*100.0);

        // if (_logFile.is_open()) {
        //     if (_firstPrint) {
        //         _logFile << "mean, mse, sd, inliers\n";
        //         _firstPrint = false;
        //     } else {
        //         _logFile << "\n";
        //     }

        //     if (inliers->indices.size() < 5){
        //         _logFile << "0.0, 0.0, 0.0, " << inliers->indices.size();
        //     } else {

        //         _logFile << std::to_string(mean_error) << ", "
        //                  << std::to_string(MSE) << ", "
        //                  << std::to_string(sigma) << ", "
        //                  << inliers->indices.size();
        //     }

        //     /*for (int i = 0; i < err.size(); i++){
        //       if (i != 0) _detailedLogFile << ", ";
        //       _detailedLogFile << std::to_string(err[i]);
        //   }
        //   _detailedLogFile << "\n";*/

        // }

        // // set up dimensions
        // featureMsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        // featureMsg.layout.dim[0].size = 4;
        // featureMsg.layout.dim[0].stride = 1;
        // featureMsg.layout.dim[0].label = "x";

        // // copy in the data
        // featureMsg.data.clear();
        // featureMsg.data.resize(4);

        // for (int j=0;j<cropped_inliers.size();j++) {
        //     if (cropped_inliers[j]->indices.size() < 5){
        //         featureMsg.left.Mean_error = 0.0;
        //         featureMsg.left.MSE = 0.0;
        //         featureMsg.left.sigma = 0.0;
        //         featureMsg.left.indices = 0.0;
        //         featureMsg.middle.Mean_error = 0.0;
        //         featureMsg.middle.MSE = 0.0;
        //         featureMsg.middle.sigma = 0.0;
        //         featureMsg.middle.indices = 0.0;
        //         featureMsg.right.Mean_error = 0.0;
        //         featureMsg.right.MSE = 0.0;
        //         featureMsg.right.sigma = 0.0;
        //         featureMsg.right.indices = 0.0;
        //     }
        // }

        // Publish error data as ROS msg
        featureMsg.left.mean_error = mean_errors[0];
        featureMsg.left.MSE = MSEs[0];
        featureMsg.left.sigma = sigmas[0];
        featureMsg.left.indices = cropped_inliers[0]->indices.size();
        featureMsg.middle.mean_error = mean_errors[1];
        featureMsg.middle.MSE = MSEs[1];
        featureMsg.middle.sigma = sigmas[1];
        featureMsg.middle.indices = cropped_inliers[1]->indices.size();
        featureMsg.right.mean_error = mean_errors[2];
        featureMsg.right.MSE = MSEs[2];
        featureMsg.right.sigma = sigmas[2];
        featureMsg.right.indices = cropped_inliers[2]->indices.size();

        _pub_feature.publish(featureMsg);

        // Publish pointclouds
        if (_enable_sending) {
            // Publish points
            //sensor_msgs::PointCloud2 cloud_publish;
            sensor_msgs::PointCloud2 cloud_publishL;
            sensor_msgs::PointCloud2 cloud_publishM;
            sensor_msgs::PointCloud2 cloud_publishR;
            sensor_msgs::PointCloud2 cloud_publish_out;
            //pcl::toROSMsg(*cloud_pub, cloud_publish);
            pcl::toROSMsg(*cloud_pub_L, cloud_publishL);
            pcl::toROSMsg(*cloud_pub_M, cloud_publishM);
            pcl::toROSMsg(*cloud_pub_R, cloud_publishR);
            pcl::toROSMsg(*cloud, cloud_publish_out);
            //cloud_publish.header = msg->header;
            cloud_publishL.header = msg->header;
            cloud_publishM.header = msg->header;
            cloud_publishR.header = msg->header;
            cloud_publish_out.header = msg->header;
            //_pub_inliers.publish(cloud_publish);
            _pub_outliers.publish(cloud_publish_out);
            _pub_croppedL.publish(cloud_publishL);
            _pub_croppedM.publish(cloud_publishM);
            _pub_croppedR.publish(cloud_publishR);
        }
    }

    void createColors(){
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        for (int i=0;i<20;i++){
            while (r<70 && g < 70 && b < 70){
                r = rand()%(255);
                g = rand()%(255);
                b = rand()%(255);
            }
            Color c(r,g,b);
            r = 0;
            g = 0;
            b = 0;
            colors.push_back(c);
        }
    }

    void spin(){
        ros::spin();
    }

private:

    // Node
    ros::NodeHandle _nh;
    std::string _name;

    // Publishers
    ros::Publisher _pub_inliers; // Display inliers for each plane
    ros::Publisher _pub_feature;  // Error publisher
    ros::Publisher _pub_croppedL; // Left segment
    ros::Publisher _pub_croppedM; // Middle segment
    ros::Publisher _pub_croppedR; // Right segment
    ros::Publisher _pub_outliers; // Outliers for the whole original pointcloud

    // Subscriber
    ros::Subscriber _subs;

    // Service
    // ros::ServiceServer _featureLoggingService;

    // Algorithm parameters
    double _max_distance;
    bool _color_pc_with_error;
    bool _enable_sending;
    bool _enable_crop;
    double total_width;
    // double minX_L;
    // double maxX_L;
    // double minX_M;
    // double maxX_M;
    // double minX_R;
    // double maxX_R;
    double minY_L;
    double maxY_L;
    double minY_M;
    double maxY_M;
    double minY_R;
    double maxY_R;
    double minZ;
    double maxZ;

    // Logging
    std::ofstream _logFile;
    //std::ofstream _detailedLogFile;
    bool _firstPrint;

    // Colors
    std::vector<Color> colors;

    // Dynamic reconfigure
    dynamic_reconfigure::Server<terrain_characterizer::algorithmParametersConfig> drServer;
    dynamic_reconfigure::Server<terrain_characterizer::algorithmParametersConfig>::CallbackType drCallback;
};

int main(int argc,char** argv){

    sleep(10);

    // Initialize ROS
    ros::init(argc,argv,"pointCloudPlaneFitter");
    ros::NodeHandle nh("~");

    pointCloudPlaneFitter pf(nh);
    pf.spin();

    return 0;
}
