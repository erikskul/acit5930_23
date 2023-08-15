#include <iostream>
#include <fstream>
#include <cmath>

#include <ros/ros.h>
#include <ros/publisher.h>
#include <sensor_msgs/PointCloud2.h>
#include <dynamic_reconfigure/server.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

// PCL
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/io/png_io.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/png_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/principal_curvatures.h>


#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <tf/transform_datatypes.h>

#include "terrain_nav/navParamsConfig.h"
#include <opencv2/opencv.hpp>



class heightMapGenerator{ 
public:
    heightMapGenerator(ros::NodeHandle nh): _nh(nh){
        initialize();
    }


    void initialize(){
        _name = ros::this_node::getName();

        // Publisher
        _pub_heightcloud_tf= _nh.advertise<sensor_msgs::PointCloud2>("terrain_nav/heighcloud_tf", 2);
        _pub_heightcloud= _nh.advertise<sensor_msgs::PointCloud2>("terrain_nav/heighcloud_original", 2);
        _pub_heightmap = _nh.advertise<sensor_msgs::Image>("terrain_nav/heightmap", 2);
        // Create publisher for the point cloud and marker for the plane
        pub_points = _nh.advertise<sensor_msgs::PointCloud2>("plane_points", 1);
        pub_marker = _nh.advertise<visualization_msgs::Marker>("plane_marker", 1);
        // Sub
        _sub = _nh.subscribe("/camera/depth/color/points",1,&heightMapGenerator::callback,this);
        
        ros::param::param<double>("~scale",scale,false);
        ros::param::param<double>("~minX",minX,false);
        ros::param::param<double>("~maxX",maxX,false);
        ros::param::param<double>("~minY",minY,false);
        ros::param::param<double>("~maxY",maxY,false);
        ros::param::param<double>("~minZ",minZ,false);
        ros::param::param<double>("~maxZ",maxZ,false);
        ros::param::param<double>("~max_distance",max_distance,false);
        ros::param::param<double>("~limitMin",limitMin,false);
        ros::param::param<double>("~limitMax",limitMax,false);
        ros::param::param<double>("~radius",radius,false);
        ros::param::param<int>("~which",which,false);

        drCallback = boost::bind( &heightMapGenerator::updateParameters, this, _1, _2);
        drServer.setCallback(drCallback);

        // Info
        ROS_INFO("%s: node initialized.",_name.c_str());
        
    }

    void updateParameters(terrain_nav::navParamsConfig& config, uint32_t level){
        scale = config.scale;
        minX = config.minX;
        maxX = config.maxX;
        minY = config.minY;
        maxY = config.maxY;
        minZ = config.minZ;
        maxZ = config.maxZ;
        max_distance = config.max_distance;
        limitMin = config.limitMin;
        limitMax = config.limitMax;
        radius = config.radius;
        which = config.which;
    }

    void callback(const sensor_msgs::PointCloud2::ConstPtr &msg){

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_msg (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg,*cloud_msg);

        // // Fix TF
        // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tf (new pcl::PointCloud<pcl::PointXYZ>);
        // cloud_tf = fixTF(cloud_msg);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped (new pcl::PointCloud<pcl::PointXYZ>);
        cloud_cropped = crop(cloud_msg);

        // // Save point cloud from crop() function
        // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped_tf (new pcl::PointCloud<pcl::PointXYZ>);
        // cloud_cropped_tf = crop(cloud_tf);

        // Get min and max values of current pointcloud
        Eigen::Vector4f cloudMin, cloudMax;
        pcl::getMinMax3D(*cloud_cropped, cloudMin, cloudMax);
        double min_x = cloudMin[0];
        double max_x = cloudMax[0];
        double min_y = cloudMin[1];
        double max_y = cloudMax[1];
        double min_z = cloudMin[2];
        double max_z = cloudMax[2];
        
        // Apply PCL voxel grid filter on cloud_cropped
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud (cloud_cropped);
        sor.setLeafSize (0.005f, 0.005f, 0.005f);
        sor.setDownsampleAllData(false);
        sor.filter (*cloud_filtered);

        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // Optional
        seg.setOptimizeCoefficients(true);

        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(1.0);
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        // Extract the planar inliers from the input cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_points (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*inlier_points);

        // Create a copy of the input cloud and extract the points that lie outside the plane
        pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_points (new pcl::PointCloud<pcl::PointXYZ>);
        extract.setNegative(true);
        extract.filter(*outlier_points);

        // set up normal estimation
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud_filtered);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch(radius); // set search radius for estimating local surface normals
        ne.compute(*cloud_normals); // compute the local surface normals at each point

        // Calculate image size ratios and scale it to the desired size
        double pc_width =  max_x - min_x;
        double pc_height = max_y - min_y;
        double z_range = max_z - min_z;
        
        // double image_width = (pc_width * scale)+1; // pixel size in x direction
        // double image_height = (pc_height * scale)+1; // pixel size in y direction
        double image_width = 16.0; // pixel size in x direction
        double image_height = 16.0; // pixel size in y direction
        int image_width_int = (int)image_width;
        int image_height_int = (int)image_height;

        // ROSINFO image size
        //ROS_INFO("Image size: %f x %f", image_width, image_height);

        // Declare heightmap limits
        float height_min = limitMin;
        float height_max = limitMax;
        // float height_min = min_z;
        // float height_max = max_z;

        cv::Mat heightmap = cv::Mat::zeros(image_width_int,image_height_int,CV_8UC1);
        for(int i=0;i<cloud_filtered->size();i++){
            // Get point values
            pcl::PointXYZ point = cloud_filtered->points[i];
            float point_x = cloud_filtered->points[i].x;
            float point_y = cloud_filtered->points[i].y;
            float point_z = cloud_filtered->points[i].z;

            // double A = cloud_normals->points[i].normal_x;
            // double B = cloud_normals->points[i].normal_y;
            // double C = cloud_normals->points[i].normal_z;

            // // Calculate the angle between the normal vector of the point and the normal vector pointing straight up (0, 0, 1)
            // double dot_product = C;
            // double normal_magnitude = sqrt(A * A + B * B + C * C);
            // double cos_theta = dot_product / normal_magnitude;
            // double slope_angle_rad = acos(cos_theta); // Result is in radians
            // double slope_angle_deg = slope_angle_rad * (180.0 / M_PI); // Convert to degrees
            // double slope_angle_deg_normalized = slope_angle_deg / 90.0;
            // unsigned char slope_value_encoded = static_cast<unsigned char>(slope_angle_deg_normalized * 255);

            // NEW with normals
            // Compute the height as the signed distance along the local surface normal from the ground plane
            // float D = coefficients->values[3];
            // float height = (-A * point_x - B * point_y - C * point_z + D) / std::sqrt(A * A + B * B + C * C);

            // Original apporach
            // Calculate depth value of the current point, scaled to 0-255
            // Problem: this is based on depth, not height. 
            // Produces a heightmap that shows terrain features in better detail, but also gradient due to distance which might not be desirable
            //int value = ((cloud_filtered->points[i].z - min_height) * 255 / height_range);
            //float height = cloud_filtered->points[i].z;

            // Calculate height value of the current point
            //float height = point_z - ((coefficients->values[0] * point_x) + (coefficients->values[1] * point_y) + coefficients->values[3]);
            double height = point2planedistance(point,coefficients);

            // Linear mapping of height value to range of 0 to 255
            int height_mapped;
            if (height <= height_min) {
                height_mapped = 0;
            } else if (height >= height_max) {
                height_mapped = 255;
            } else {
                height_mapped = (int)((height - height_min) * 255 / (height_max - height_min));
            }

            // Calculate x and y position in heightmap of the current point
            int x = (((point_x - min_x) / pc_width) * (image_width_int - 1));
            int y = (((point_y - min_y) / pc_height) * (image_height_int - 1));
            // ROS_INFO("x: %d, y: %d, value: %d",x,y,value);
    
            // Set heightmap pixel value
            heightmap.at<uchar>(y,x) = height_mapped;

            // 3channel
            // heightmap.at<cv::Vec3b>(y,x)[0] = height_mapped;
            // heightmap.at<cv::Vec3b>(y,x)[1] = slope_value_encoded;
            // heightmap.at<cv::Vec3b>(y,x)[2] = 0;

            // ROS_INFO the z value, height and height_mapped
            // ROS_INFO("z: %f, height: %f, height_mapped: %d",point_z,height,height_mapped);

        }

        // Rotate heightmap 90 degrees clockwise
        // cv::rotate(heightmap, heightmap, cv::ROTATE_90_CLOCKWISE);
        // Flip heightmap vertically
        //cv::flip(heightmap, heightmap, 1);
        
        // Save heightmap
        // cv::imwrite("heightmap.png",heightmap);
        // ROS_INFO("heightmap saved");

        // // PC2 cropped_tf_filtered Publisher
        // sensor_msgs::PointCloud2 heightcloud_tf;
        // pcl::toROSMsg(*cloud_filtered, heightcloud_tf);
        // heightcloud_tf.header = msg->header;
        // _pub_heightcloud_tf.publish(heightcloud_tf);

        // PC2 cropped Publisher
        sensor_msgs::PointCloud2 heightcloud;
        pcl::toROSMsg(*cloud_filtered, heightcloud);
        heightcloud.header = msg->header;
        _pub_heightcloud.publish(heightcloud);

        // Heightmap ROS Publisher
        cv_bridge::CvImage img_bridge;
        sensor_msgs::Image img_msg;
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_8UC1, heightmap);
        img_bridge.toImageMsg(img_msg);
        _pub_heightmap.publish(img_msg);
        
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr crop(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_msg){
        // Crop PCL point cloud to a box of certain size
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::CropBox<pcl::PointXYZ> cropFilter;
        cropFilter.setInputCloud(cloud_msg);
        cropFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
        cropFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
        cropFilter.filter(*cloud_cropped);

        return cloud_cropped;

    }

    double point2planedistance(pcl::PointXYZ pt, pcl::ModelCoefficients::Ptr coefficients){
        double f1 = fabs(coefficients->values[0]*pt.x+coefficients->values[1]*pt.y+coefficients->values[2]*pt.z+coefficients->values[3]);
        double f2 = sqrt(pow(coefficients->values[0],2)+pow(coefficients->values[1],2)+pow(coefficients->values[2],2));
        return f1/f2;
    }    

    void spin(){
        ros::spin();
    }

private:

    ros::NodeHandle _nh;
    std::string _name;

    // Publisher
    ros::Publisher _pub_heightcloud_tf;
    ros::Publisher _pub_heightcloud;
    ros::Publisher _pub_heightmap;
    ros::Publisher pub_points;
    ros::Publisher pub_marker;
    // Subscriber
    ros::Subscriber _sub;

    // Dynamic reconfigure
    dynamic_reconfigure::Server<terrain_nav::navParamsConfig> drServer;
    dynamic_reconfigure::Server<terrain_nav::navParamsConfig>::CallbackType drCallback;

    double scale;
    double minX;
    double maxX;
    double minY;
    double maxY;
    double minZ;
    double maxZ;
    double max_distance;
    double limitMin;
    double limitMax;
    double radius;
    int which;

};

int main(int argc,char** argv){

    sleep(1);

    // Initialize ROS
    ros::init(argc,argv,"heightMapGenerator");
    ros::NodeHandle nh("~");

    heightMapGenerator hg(nh);
    hg.spin();

    return 0;
}


// // Perform MLS smoothing on the input cloud
//   pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
//   mls.setInputCloud (cloud);
//   mls.setSearchRadius (0.1);
//   mls.setPolynomialOrder (2);
//   mls.setUpsamplingMethod (pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
//   mls.setUpsamplingRadius (0.2);
//   pcl::PointCloud<pcl::PointXYZ>::Ptr smoothed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
//   mls.process (*smoothed_cloud);