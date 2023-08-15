#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem>

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

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/String.h>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <tf/transform_datatypes.h>
#include <opencv2/opencv.hpp>

pcl::PointCloud<pcl::PointXYZ>::Ptr crop(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_msg){
    // Crop PCL point cloud to a box of certain size
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::CropBox<pcl::PointXYZ> cropFilter;
    cropFilter.setInputCloud(cloud_msg);
    cropFilter.setMin(Eigen::Vector4f(-0.8, -0.5, -2.5, 1.0));
    cropFilter.setMax(Eigen::Vector4f(0.8, 0.1, 2.5, 1.0));
    cropFilter.filter(*cloud_cropped);

    return cloud_cropped;

}

double point2planedistance(pcl::PointXYZ pt, pcl::ModelCoefficients::Ptr coefficients){
    double f1 = fabs(coefficients->values[0]*pt.x+coefficients->values[1]*pt.y+coefficients->values[2]*pt.z+coefficients->values[3]);
    double f2 = sqrt(pow(coefficients->values[0],2)+pow(coefficients->values[1],2)+pow(coefficients->values[2],2));
    return f1/f2;
}

int main(int argc,char** argv){

    sleep(1);

    // Print that node has started
    std::cout << "Node has started" << std::endl;
    
    // Set the path to the directory containing ROS bag files
    std::string bag_dir_path = "/media/user/Plex/ROSBAGS/day1_1";
    std::string heightmap_dir_path = "/home/user/scout_ws/datasetAll/images";
    std::string csv_path = "/home/user/scout_ws/datasetAll/bag_to_image.csv";
    int scale = 70;
    double limits = 0.1; // 0.1 for euclidian, 1.5 for normals

    std::ifstream csv_file(csv_path);

    if (!csv_file.is_open()){
        std::cerr << "Failed to open CSV file: " << csv_path << std::endl;
        return -1;
    }

    std::string line;
    std::vector<std::string> bag_files;
    std::vector<std::string> image_names;
    std::vector<bool> mirrored;

    // while (std::getline(csv_file, line)){
    //     // Split the line by comma delimiter to get the bag file name (first column)
    //     size_t delimiter_pos = line.find(",");
    //     if (delimiter_pos != std::string::npos){
    //         std::string bag_file = line.substr(0, delimiter_pos);
    //         std::string image_name = line.substr(delimiter_pos + 1);
    //         // I keep getting whitespace at the end of image_name from CSV file, could not figure out why, so I just remove it here
    //         image_name.pop_back();
    //         bag_files.push_back(bag_file);
    //         image_names.push_back(image_name);
    //     }
    // }

    while (std::getline(csv_file, line)){
    // Split the line by comma delimiter to get the bag file name (first column)
        size_t first_delimiter_pos = line.find(",");
        if (first_delimiter_pos != std::string::npos){
            std::string bag_file = line.substr(0, first_delimiter_pos);
            bag_files.push_back(bag_file);

            // Find the second comma to split the rest of the string
            size_t second_delimiter_pos = line.find(",", first_delimiter_pos + 1);
            if (second_delimiter_pos != std::string::npos) {
                std::string image_name = line.substr(first_delimiter_pos + 1, second_delimiter_pos - first_delimiter_pos - 1);
                
                // I keep getting whitespace at the end of image_name from CSV file, could not figure out why, so I just remove it here
                if (!image_name.empty() && isspace(image_name.back())) {
                    image_name.pop_back();
                }

                // Check the third column for "mirrored"
                std::string third_column = line.substr(second_delimiter_pos + 1);
                bool is_mirrored = (third_column.find("mirrored") != std::string::npos);

                
                image_names.push_back(image_name);
                mirrored.push_back(is_mirrored);
            }

            else {
                
                std::string image_name = line.substr(first_delimiter_pos + 1);
                if (!image_name.empty() && isspace(image_name.back())) {
                    image_name.pop_back();
                }
                image_names.push_back(image_name);
                mirrored.push_back(false);

            }
        }
    }

    // Iterate over each bag file and process it
    for (int i = 0; i < bag_files.size(); i++){
        std::string image_name = image_names[i];
        std::string bag_file_path = bag_dir_path + "/" + bag_files[i];
        std::string image_path = heightmap_dir_path + "/" + image_name;
        bool is_mirrored = mirrored[i];

        // print all of the valiables above in one ROS_INFO
        //ROS_INFO("Processing bag file: %s, image name: %s, image path: %s, is mirrored: %d", bag_file_path.c_str(), image_name.c_str(), image_path.c_str(), is_mirrored);


        // flips an image vertically if a mirror tag is present. it assumes that the previous image is already converted, and just flips that one
        if (is_mirrored) {
            // remove the .png extension
            std::string temp_name = image_name;
            temp_name.pop_back();
            temp_name.pop_back();
            temp_name.pop_back();
            temp_name.pop_back();
            
            // convert image_name to int, and then substract 1, and convert that back to string
            int prev_number = std::stoi(temp_name);
            prev_number--;
            std::string prev_image = std::to_string(prev_number);

            // load the previous image
            cv::Mat prev_image_mat = cv::imread(heightmap_dir_path + "/" + prev_image + ".png", cv::IMREAD_GRAYSCALE);

            // flip the image vertically
            cv::flip(prev_image_mat, prev_image_mat, 1);

            // save the flipped image
            cv::imwrite(image_path, prev_image_mat);
        }

        else {


            // Check if the file exists
            std::ifstream file_check(bag_file_path);
            if (!file_check) {
                std::cerr << "Rosbag file " << bag_file_path << " does not exist. Skipping..." << std::endl;
                continue;
            }
            file_check.close();

            rosbag::Bag bag;
            bag.open(bag_file_path, rosbag::bagmode::Read);

            // Get the first message of the topic /points
            std::vector<std::string> topics;
            topics.push_back(std::string("/camera/depth/color/points"));
            rosbag::View view(bag, rosbag::TopicQuery(topics));

            // PointCloud2 object to store the pointcloud
            sensor_msgs::PointCloud2::ConstPtr msg;

            // Get the first msg of the topic
            BOOST_FOREACH(rosbag::MessageInstance const m, view){
                msg = m.instantiate<sensor_msgs::PointCloud2>();
                if (msg != NULL){
                    break;
                }
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_msg (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg,*cloud_msg);

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped (new pcl::PointCloud<pcl::PointXYZ>);
            cloud_cropped = crop(cloud_msg);

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

            // // Extract the planar inliers from the input cloud
            // pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_points (new pcl::PointCloud<pcl::PointXYZ>);
            // pcl::ExtractIndices<pcl::PointXYZ> extract;
            // extract.setInputCloud(cloud_filtered);
            // extract.setIndices(inliers);
            // extract.setNegative(false);
            // extract.filter(*inlier_points);

            // // Create a copy of the input cloud and extract the points that lie outside the plane
            // pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_points (new pcl::PointCloud<pcl::PointXYZ>);
            // extract.setNegative(true);
            // extract.filter(*outlier_points);

            // set up normal estimation
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
            ne.setInputCloud(cloud_filtered);
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
            ne.setSearchMethod(tree);
            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
            ne.setRadiusSearch(0.03); // set search radius for estimating local surface normals
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
            float height_min = -0.03;
            float height_max = 0.075;
            // float height_min = min_z;
            // float height_max = max_z;

            cv::Mat heightmap = cv::Mat::zeros(image_width_int,image_height_int,CV_8UC1);
            //cv::Mat heightmap = 255 * cv::Mat::ones(image_width_int,image_height_int,CV_8UC1);
            for(int i=0;i<cloud_filtered->size();i++){
                // Get point values
                pcl::PointXYZ point = cloud_filtered->points[i];
                float point_x = cloud_filtered->points[i].x;
                float point_y = cloud_filtered->points[i].y;
                float point_z = cloud_filtered->points[i].z;

                // double A = cloud_normals->points[i].normal_x;
                // double B = cloud_normals->points[i].normal_y;
                // double C = cloud_normals->points[i].normal_z;

                // Calculate the angle between the normal vector of the point and the normal vector pointing straight up (0, 0, 1)
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
                //ROS_INFO("x: %d, y: %d, value: %d",x,y,static_cast<int>(height));
        
                // Set heightmap pixel value
                heightmap.at<uchar>(y,x) = height_mapped;

                // // 3channel
                // heightmap.at<cv::Vec3b>(y,x)[0] = height_mapped;
                // heightmap.at<cv::Vec3b>(y,x)[1] = slope_value_encoded;
                // heightmap.at<cv::Vec3b>(y,x)[2] = 0;

                // ROS_INFO the z value, height and height_mapped
                // ROS_INFO("z: %f, height: %f, height_mapped: %d",point_z,height,height_mapped);

            }

            // // Rotate heightmap 90 degrees clockwise
            // cv::rotate(heightmap, heightmap, cv::ROTATE_90_CLOCKWISE);
            // // Flip heightmap vertically
            // cv::flip(heightmap, heightmap, 1);
            
            // Save heightmap
            cv::imwrite(image_path,heightmap);
        }
        
    // ROSINFO of status of how many bags have been processed
    ROS_INFO("Processed bag %d of %d: %s", i+1, static_cast<int>(bag_files.size()), bag_file_path.c_str());

    }

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