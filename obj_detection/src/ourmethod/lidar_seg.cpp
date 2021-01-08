#include "lidar_seg.h"
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

lidar_seg::lidar_seg(double pan_res, double tilt_res){

  get_cons(pan_res, tilt_res);
  //Camera rotation matrix
  for (int i = 0; i < 3; i++)
  {
      for (int j = 0; j < 3; j++)
      {
          this->DepthtoRGBMat(i,j)= RDepthAxisToRgbAxis[3*i+j];
      }
  }
  
  //Camera translation matrix
  for (int i = 0; i < 3; i++)
  {
      this->DepthtoRGBMat(i,3) = TDepthAxisToRgbAxis[i];
  }
}
void lidar_seg::get_cons( double pan_res, double tilt_res){


  this->pan_resolution = pan_res;
  this->tilt_resolution = tilt_res;

  index_vec.clear();

}

void lidar_seg::set_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr  in_cloud ){



  get_cloud(in_cloud);

}

/*
Author: M.U
Date: 11/18/2020
Method: This method sets parameters of the class with depth image.
*/
void lidar_seg::set_cloud( cv::Mat depth) {


    visited.clear();
    clusters.clear();

    visited.resize(depth.rows*depth.cols);
    clusters.resize(depth.rows * depth.cols);
    std::fill(visited.begin(), visited.end(), false); //visited indexes
    std::fill(clusters.begin(), clusters.end(), 0); //clusters


    total_pan = round(360 / this->pan_resolution) + 10; //10 is added for the safety of the code
    total_tilt = round(360 / this->tilt_resolution) + 10; // 10 is added for the safety of the code
    range_image = Eigen::MatrixXd::Zero(total_pan, total_tilt); //range iimage data collection Initializion

}

void lidar_seg::get_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr  in_cloud){

  copyPointCloud(*in_cloud, *cloud);
  //cloud = in_cloud; //take input cloud
  visited.clear();
  clusters.clear();

  visited.resize(cloud->size());
  clusters.resize(cloud->size());
  std::fill(visited.begin(), visited.end(), false); //visited indexes
  std::fill(clusters.begin(), clusters.end(), 0); //clusters
  //  cout<<"size:"<<cloud->size()<<endl;

  total_pan = round(360 /this->pan_resolution); //10 is added for the safety of the code
  total_tilt = round(180/this->tilt_resolution); // 10 is added for the safety of the code
  range_image = Eigen::MatrixXd::Zero(total_pan,total_tilt); //range iimage data collection Initializion
  //cout<<range_image<<endl;


}


void lidar_seg::set_params(double pan_eps, double tilt_eps, double dist_eps, int min_points){

  get_params(pan_eps,tilt_eps,dist_eps, min_points);

}

void lidar_seg::get_params(double pan_eps, double tilt_eps, double dist_eps, int min_points){

  //taking the parameters
  this->pan_epsilon = pan_eps;
  this->tilt_epsilon = tilt_eps;
  this->dist_epsilon = dist_eps;
  this->seg_min = min_points;


  pan_direction = round(this->pan_epsilon/ this->pan_resolution); //find the maximum direction
  tilt_direction = round(this->tilt_epsilon / this->tilt_resolution); //find the maximum direction


}

void lidar_seg::to_sphere() {

    // pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_object(new pcl::PointCloud<pcl::PointXYZ>);


    sphere_cloud.points.resize(cloud->points.size());


    pcl::PointXYZ sphere_points;

    index_vec.resize(total_pan + 10, std::vector<int>(total_tilt + 10));
    std::fill(index_vec.begin(), index_vec.end(), vector<int>(total_tilt + 10, -1)); //index vector is filled with -1


    int pan_idx, tilt_idx;


    // To sphere
    for (int i = 0; i < cloud->size(); i++) {

        sphere_cloud.points[i].x = sqrt(cloud->points[i].x * cloud->points[i].x + cloud->points[i].y * cloud->points[i].y + cloud->points[i].z * cloud->points[i].z); //distance
        sphere_cloud.points[i].y = asin(cloud->points[i].z / sphere_cloud.points[i].x); //tilt angle
        sphere_cloud.points[i].z = atan2(cloud->points[i].x, cloud->points[i].y); //pan angle

        if (sphere_cloud.points[i].z < 0.0)
        {
            sphere_cloud.points[i].z = sphere_cloud.points[i].z + 2 * M_PI; //0 and 2*M_PI
        }


        sphere_cloud.points[i].y = sphere_cloud.points[i].y + M_PI / 2; //0 and M_PI


        sphere_cloud.points[i].y = sphere_cloud.points[i].y * 180.0 / M_PI; //to degree
        sphere_cloud.points[i].z = sphere_cloud.points[i].z * 180.0 / M_PI; //to degree





        pan_idx = round(sphere_cloud.points[i].z / this->pan_resolution); //finding pan index
        tilt_idx = round(sphere_cloud.points[i].y / this->tilt_resolution); //finding tilt index


        //cout<<sphere_cloud.points[i].y<<endl;
        index_vec[pan_idx][tilt_idx] = i; //take an index vec;

    }


    //   sphere_cloud = sphere_object;

}


/*
Author: M.U
Date: 11/11/2020
Method: This method generates point cloud expressed in spherical coo. form an input 2-D depth Matrix expressed wrt pan and titl angles.
*/


void lidar_seg::to_sphere(cv::Mat depth, cv::Mat rgb) {

    sphere_cloud.points.resize(depth.rows*depth.cols);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    tempCloud->points.resize(depth.rows * depth.cols);
    index_vec.resize(total_pan + 10, std::vector<int>(total_tilt + 10));
    std::fill(index_vec.begin(), index_vec.end(), vector<int>(total_tilt + 10, -1)); //index vector is filled with -1

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    source_cloud->resize(1);

    int pan_idx, tilt_idx;
    int index = 0;
    for (int i = 0; i < depth.rows; i++)
    {
        for (int j = 0; j < depth.cols; j++)
        {
            unsigned short p = depth.at<unsigned short>(i, j);
            float fov_vertical = -(TILT_RANGE / 2) + i * (TILT_RANGE / depth.rows);
            float fov_horizontal = PAN_RANGE / 2 - j * (PAN_RANGE / depth.cols);
            float x_multiplier = (j - DEPTH_C_X) / DEPTH_FOCAL_X;
            float y_multiplier = (i- DEPTH_C_Y) / DEPTH_FOCAL_Y;
            
            // Get 3D coordinates
            float z_world = double(p)/1000 / sqrt(x_multiplier * x_multiplier + y_multiplier * y_multiplier + 1); 
            float x_world = z_world * x_multiplier;
            float y_world = z_world * y_multiplier;
            
            tempCloud->points[index].x = x_world;
            tempCloud->points[index].y = y_world;
            tempCloud->points[index].z = z_world;


            source_cloud->points[0].x = tempCloud->points[index].x;
            source_cloud->points[0].y = tempCloud->points[index].y;
            source_cloud->points[0].z = tempCloud->points[index].z;

            pcl::transformPointCloud((*source_cloud), *transformed_cloud, this->DepthtoRGBMat);

            tempCloud->points[index].x = transformed_cloud->points[0].x;
            tempCloud->points[index].y = transformed_cloud->points[0].y;
            tempCloud->points[index].z = transformed_cloud->points[0].z;

            x_world = tempCloud->points[index].x;
            y_world = tempCloud->points[index].y;
            z_world = tempCloud->points[index].z;

            // Get corresponding RGB pixels
            float x_rgbcam = RGB_FOCAL_X * x_world /z_world + RGB_C_X;
            float y_rgbcam = RGB_FOCAL_Y * y_world / z_world + RGB_C_Y;

            // "Interpolate" pixel coordinates 
            int px_rgbcam = cvRound(x_rgbcam);
            int py_rgbcam = cvRound(y_rgbcam);
 
            //assign color if applicable
            if (px_rgbcam > 0 && px_rgbcam < depth.cols && py_rgbcam>0 && py_rgbcam < depth.rows)
            {
                tempCloud->points[index].b = rgb.at<cv::Vec3b>(py_rgbcam, px_rgbcam)[0];
                tempCloud->points[index].g = rgb.at<cv::Vec3b>(py_rgbcam, px_rgbcam)[1];
                tempCloud->points[index].r = rgb.at<cv::Vec3b>(py_rgbcam, px_rgbcam)[2];
            }
            else
            {
                tempCloud->points[index].b = 255;
                tempCloud->points[index].g = 255;
                tempCloud->points[index].r = 255;

            }

            //convert to spherical coordinate system
            sphere_cloud.points[index].x = sqrt(x_world * x_world + y_world * y_world + z_world * z_world);
            if (sphere_cloud.points[index].x == 0)
            {
                sphere_cloud.points[index].x = 10000;
            }

            sphere_cloud.points[index].y = asin(z_world /(sphere_cloud.points[index].x));
            sphere_cloud.points[index].z = atan2(y_world, x_world);

            if (sphere_cloud.points[index].z < 0.0)
            {
                sphere_cloud.points[index].z = sphere_cloud.points[index].z + 2 * M_PI; //0 and 2*M_PI
            }

           sphere_cloud.points[index].y = sphere_cloud.points[index].y + M_PI / 2; //0 and M_PI


            sphere_cloud.points[index].y = sphere_cloud.points[index].y * 180.0 / M_PI; //to degree
            sphere_cloud.points[index].z = sphere_cloud.points[index].z * 180.0 / M_PI; //to degree


            pan_idx = round(sphere_cloud.points[index].z / this->pan_resolution); //finding pan index
            tilt_idx = round(sphere_cloud.points[index].y / this->tilt_resolution); //finding tilt index

            index_vec[pan_idx][tilt_idx] = index; //take an index vec;
            ++index;
            ++p;
        }
    }
    cloud = tempCloud;
    

}


void::lidar_seg::segment(){


  to_sphere(); //find spherical coordinates and range images
  //  cout<<"size:"<<sphere_cloud.size()<<endl;

  int k ; //while iterator to expand the cluster
  for(int i = 0; i< sphere_cloud.points.size(); i++){

    if(visited[i] == true){

      continue; //if the value visited the algorithm skip this point
    }
    else{

      std::vector<int> neighs =  region_query(i,true); //Find the core point

      if(neighs.size()<=  0)
      {

        //if this is not a core point continue to find a core point

        continue;

      }

      else{
        //if the core point is found, the cluster will expanded with other
        cluster = cluster +1;
        clusters[i] = cluster;
        k = 0;
        visited[i] = true;
        while(k< neighs.size())
        {

        //  clusters[neighs[k]] = cluster;
          std::vector<int> neighs_expand = region_query(neighs[k],false);
          if(neighs_expand.size()> 0)
          {
            neighs.insert( neighs.end(), neighs_expand.begin(), neighs_expand.end() );
          }

          k = k + 1 ;

        }


      }

      if(neighs.size() +1 < this->seg_min){
clusters[i]= -1;
for(auto a: neighs)
  clusters[a] = -1;
  cluster = cluster -1;

      }

      neighs.clear();
    }



  }
  total_cluster= cluster;
  cout<<"cluster: "<<cluster<<endl;

}

/*
Author: Mirhan U.
Date: 11/28/2020
Method: Depth data segmentation

*/
void::lidar_seg::segment(cv::Mat depth, cv::Mat rgb) {
    

    to_sphere(depth, rgb); //find spherical coordinates and range images
    int k; //while iterator to expand the cluster
    for (int i = 0; i < sphere_cloud.points.size(); i++) {

        if (visited[i] == true) {

            continue; //if the value visited the algorithm skip this point
        }
        else {

            std::vector<int> neighs = region_query(i, true); //Find the core point

            
            if (neighs.size() <= 0)
            {

                //if this is not a core point continue to find a core point

                continue;

            }

            else {
                //if the core point is found, the cluster will expanded with other
                cluster = cluster + 1;
                clusters[i] = cluster;
                k = 0;
                visited[i] = true;
              
               
                while (k < neighs.size())
                {

                    //  clusters[neighs[k]] = cluster;
                    std::vector<int> neighs_expand = region_query(neighs[k], false);


                    if (neighs_expand.size() > 0)
                    {
                        neighs.insert(neighs.end(), neighs_expand.begin(), neighs_expand.end());
                    }
                    k = k + 1;

                }
 
                

            }

            if (neighs.size() + 1 < this->seg_min) {
                clusters[i] = -1;
                for (auto a : neighs)
                    clusters[a] = -1;
                cluster = cluster - 1;

                
            }
            neighs.clear();
        }



    }
    total_cluster = cluster;
    cout << "cluster: " << cluster << endl;

}


void lidar_seg::dist(int index, int core_index, std::vector<int> &result_neigh){

      float ellipsoid_dist; //ellipsoid distance


      float pan_neigh, tilt_neigh, dist_neigh;
      float pan_dist, tilt_dist, dist_dist;

      pan_neigh = sphere_cloud.points[index].z ;
      tilt_neigh = sphere_cloud.points[index].y;
      dist_neigh = sphere_cloud.points[index].x;


      pan_dist = 180-fabs(fabs(sphere_cloud.points[core_index].z - pan_neigh) -180); //calculate pan distance between two points
      tilt_dist = sphere_cloud.points[core_index].y - tilt_neigh; // calculate tilt distance between two points
      dist_dist = dist_neigh - sphere_cloud.points[core_index].x; // calculate distance
      int red_dist = cloud->points[index].r- cloud->points[core_index].r;
      int green_dist = cloud->points[index].g - cloud->points[core_index].g;
      int blue_dist = cloud->points[index].b - cloud->points[core_index].b;
      int color_dist = red_dist * red_dist + green_dist * green_dist + blue_dist * blue_dist;

      ellipsoid_dist = ((pan_dist * pan_dist) / (this->pan_epsilon * this->pan_epsilon)) + ((tilt_dist * tilt_dist)/(this->tilt_epsilon * this->tilt_epsilon)) + ((dist_dist * dist_dist)/(this->dist_epsilon * this->dist_epsilon)); //calculate ellipsoid distance
      if((ellipsoid_dist <= 0.3) || (ellipsoid_dist <= 0.5 && color_dist<50)||(ellipsoid_dist <= 1 && color_dist < 10))
      {

        if(clusters[index]==0 || clusters[index]==cluster){
      

          min_cluster_var = min_cluster_var + 1;
          if(visited[index] == false )
          { //only unique indexes can be added
            result_neigh.push_back(index);
            visited[index] = true;
        
          }

        }

      }

}





std::vector<int> lidar_seg::region_query(int iterator, bool core){

  std::vector<int> result_vec;

  min_cluster_var = 0;

  int pan_idx = int(sphere_cloud.points[iterator].z/this->pan_resolution);
  int tilt_idx = int(sphere_cloud.points[iterator].y / this->tilt_resolution);



  int dec_idx_pan=0;
  int inc_idx_pan= 0;
  int inc_idx_tilt =0;
  int dec_idx_tilt =0;

  int idx ;
  /*
  Find possible directions in the range image
  Tilt direction can be access the boundries, for this reason the directions are splitted into 3 parts. The first part is limited only the incremental direction in tilt,
  The second one is limited in the decreasing direction in tilt,
  There is no limitation for  the third one.


  */
  for(int i = 0; i< this->tilt_direction; i++){
    inc_idx_tilt = tilt_idx + i;
    dec_idx_tilt = tilt_idx - i;


    for(int k = 0; k< this->pan_direction; k++){

      if(k==0 && i==0) // core point continue
      {
        continue;
      }

      inc_idx_pan = (pan_idx +k) % total_pan ;
      dec_idx_pan =  pan_idx - k ;
      if(dec_idx_pan<0)
      {
        dec_idx_pan = dec_idx_pan + total_pan ; // function mod
      }


      if(inc_idx_tilt<= total_tilt && dec_idx_tilt <0 ){

        //cout<<"hey"<<endl;

        idx = index_vec[inc_idx_pan][inc_idx_tilt];
        if(idx!=-1)
            dist(idx,iterator, result_vec);

        idx = index_vec[dec_idx_pan][inc_idx_tilt];
        if(idx!=-1)
            dist(idx, iterator, result_vec);



      }
      else if(inc_idx_tilt> total_tilt && dec_idx_tilt > 0)
      {


        idx = index_vec[inc_idx_pan][dec_idx_tilt] ;
        if(idx!=-1)
            dist(idx, iterator,result_vec);

        idx = index_vec[dec_idx_pan][dec_idx_tilt];
        if(idx!=-1)
            dist(idx,iterator,result_vec);


        //  cout<<"hey"<<endl;


      }

      else if(inc_idx_tilt<=total_tilt && dec_idx_tilt>=0)
      {
        //  cout<<"hey"<<endl;



        idx = index_vec[inc_idx_pan][dec_idx_tilt] ;
        if(idx!=-1)
            dist(idx, iterator,result_vec);

        idx = index_vec[dec_idx_pan][dec_idx_tilt];
        if(idx!=-1)
            dist(idx,iterator,result_vec);



        idx = index_vec[inc_idx_pan][inc_idx_tilt];
        if(idx!=-1)
            dist(idx,iterator, result_vec);

        idx = index_vec[dec_idx_pan][inc_idx_tilt];
        if(idx!=-1)
            dist(idx, iterator, result_vec);

      }


    }
  }

  //Control the density
  if(min_cluster_var < seg_min){
    //if the point is not a core point or this is not a dense region, all points in the region free
    visited[iterator] = true;
    for(size_t not_core_it = 0; not_core_it < result_vec.size(); not_core_it++){

      visited[result_vec[not_core_it]] = false ;

    }
    result_vec.clear();
    min_cluster_var = 0;
    return result_vec;

  }

  else
  {
    //if this is a core point, take the neighs

    for(size_t cluster_assign=0; cluster_assign<result_vec.size(); cluster_assign++){
      if(core == true){
        clusters[result_vec[cluster_assign]] = cluster +1; //assign the cluster

      }

      else{
        clusters[result_vec[cluster_assign]] = cluster; //assign the cluster

      }
    }
    min_cluster_var = 0;

    return result_vec;

  }




}






void lidar_seg::take_colored_cloud(pcl::PointCloud<pcl::PointXYZRGB> &colored_cloud){

    vector<size_t> color_vec(total_cluster);

    for(int i = 0;i<total_cluster;i++){
    // srand(time(NULL));
    size_t randNum = std::rand()%(256) ;

    color_vec[i] = randNum;
    }

    colored_cloud.resize(clusters.size());


    vector<vector<int> > color_vector;
      //color_vector.resize(cluster,vector<uint8_t>(3));
    vector<int> rgb;
    int c1,c2,c3;
    for(size_t color_cluster= 0 ; color_cluster<cluster; color_cluster++){

        c1 = rand() % 255 ;
        c2 = rand() % 255 ;
        c3 = rand() % 255 ;

        rgb.push_back(c1);
        rgb.push_back(c2);
        rgb.push_back(c3);

        color_vector.push_back(rgb);
        rgb.clear();
    }





    for(size_t color_it = 0; color_it< clusters.size(); color_it++){

      if(clusters[color_it] <= 0 ) //non-clustered points are skipped
      {



        continue;

      }

      colored_cloud.points[color_it].x = cloud->points[color_it].x;
      colored_cloud.points[color_it].y = cloud->points[color_it].y;
      colored_cloud.points[color_it].z = cloud->points[color_it].z;

      colored_cloud.points[color_it].r = color_vector[clusters[color_it]-1][0];//(255/cluster) * clusters[color_it] ;
      colored_cloud.points[color_it].g = color_vector[clusters[color_it]-1][1];//(255/cluster) * clusters[color_it] ;
      colored_cloud.points[color_it].b = color_vector[clusters[color_it]-1][2]; //(255/cluster) * clusters[color_it] ;

    }

    //pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    //viewer.showCloud(cloud);
    //cv::waitKey(900000);




}



void lidar_seg::boundingbox(){





}
