/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 
 * Modified my Sim K 
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

const double INIT_WEIGHT = 1.0;
const int NUM_PARTICLES = 100;
const double EPS = 0.0001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    
  // Return if already initialized.
  if (is_initialized) {
    return;
  }
   // Set the number of particles
  this->num_particles = NUM_PARTICLES;
  
  // Get Standard Deviation
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  
  random_device rand_dev;
  mt19937 gen_rand(rand_dev());
  
  // Normal Distribution
  normal_distribution<double> norm_dist_x(x, std_x);
  normal_distribution<double> norm_dist_y(y, std_y);
  normal_distribution<double> norm_dist_theta(theta, std_theta);
  
  this->particles.resize(this->num_particles);
  int i = 0;
   // Generate particles.
  for (auto& particle: this->particles) {

    particle = {
      i++,
      norm_dist_x(gen_rand),
      norm_dist_y(gen_rand),
      norm_dist_theta(gen_rand),
      INIT_WEIGHT
    };

    this->weights.push_back(INIT_WEIGHT);
    this->particles.push_back(particle);
  }
  // Set the flag to true.
  this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   
    // Get Standard Deviation
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  
  random_device rand_dev;
  mt19937 gen_rand(rand_dev());
  
  // Normal Distribution
  normal_distribution<double> norm_dist_x(0, std_x);
  normal_distribution<double> norm_dist_y(0, std_y);
  normal_distribution<double> norm_dist_theta(0, std_theta);
  
  this->particles.resize(this->num_particles);
  

  // Generate particles.
  for (auto& particle: particles) {
	  
	double theta = particle.theta;
	double vel_del = velocity * delta_t;
	double yaw_del = yaw_rate * delta_t;
	

    if ( fabs(yaw_rate) < EPS ) { // No left or right direction to the motion, going straight.
      particle.x += vel_del * cos(theta);
      particle.y += vel_del * sin(theta);

    } else {
      particle.x += velocity/yaw_rate * (sin(theta + yaw_del)-sin(theta));
      particle.y += velocity/yaw_rate * (cos(theta)-cos(theta + yaw_del));
      particle.theta += yaw_del;
    }
	
	// Adding sensor noise.
    particle.x += norm_dist_x(gen_rand);
    particle.y += norm_dist_y(gen_rand);
    particle.theta += norm_dist_theta(gen_rand);
	
	
  }
   
   

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(auto& observation: observations){
		
	double min_dist = numeric_limits<double>::max();
	int map_id = -100;
		
	for(const auto& prediction: predicted){
			
		double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
		if( min_dist > distance){
		min_dist = distance;
		map_id = prediction.id;
		}
	}
	observation.id = map_id;
  }	

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) { 	
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
		
  for(auto& particle: particles){
		
	particle.weight = INIT_WEIGHT;

	// Step 1: Valid Landmarks
	vector<LandmarkObs> landmarks_in_range;
		
	for(const auto& landmark: map_landmarks.landmark_list){
		double distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
		  
		// landmark within sensor range
		if( distance < sensor_range){ 
		landmarks_in_range.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
		}
	}

	// Step 2: Transform Observations to map co-ordinates
	vector<LandmarkObs> map_observations;
	double cos_theta = cos(particle.theta);
	double sin_theta = sin(particle.theta);

	for(const auto& observation: observations){
		LandmarkObs map_observation;
		map_observation.x = (observation.x * cos_theta) - (observation.y * sin_theta) + particle.x;
		map_observation.y = (observation.x * sin_theta) + (observation.y * cos_theta) + particle.y;

		map_observations.push_back(map_observation);
	}

	// Step 3: Associate observation to landmark
	dataAssociation(landmarks_in_range, map_observations);
		
		
	// Uncertainty of measurement of landmark in x,y direction. varx and vary. 
	double std_landmark_range = std_landmark[0];
	double std_landmark_bearing = std_landmark[1];

	// Step 4: Observation comparision w.r.t particle. Calculate weights accordingly. 
	for(const auto& map_observation: map_observations){

		Map::single_landmark_s landmark = map_landmarks.landmark_list.at(map_observation.id-1);
		double dX = map_observation.x - landmark.x_f;
		double dY = map_observation.y - landmark.y_f;
		  
		double temp_x = pow(dX, 2) / (2 * pow(std_landmark_range, 2));
		double temp_y = pow(dY, 2) / (2 * pow(std_landmark_bearing, 2));
		double weight = exp(-(temp_x + temp_y)) / (2 * M_PI * std_landmark_range * std_landmark_bearing);
		particle.weight *=  weight;
	}

	this->weights.push_back(particle.weight);

  }

}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> resampled_particles;
  random_device rand_dev;
  mt19937 gen_rand(rand_dev());
  discrete_distribution<> index_dist(this->weights.begin(), this->weights.end());

  // Resample
  for(int i=0; i< this->num_particles; i++){
    int index = index_dist(gen_rand);
    resampled_particles.push_back(this->particles[index]);
  }
  this->particles = resampled_particles;
  weights.clear();


}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}