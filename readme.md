Package: object_detection

Object localisation: detects groups of objects on a flat surface and returns the location and size of the group.

Object detection: recognises models in the scene and returns their pose if the model was found. 

Red dot: gives back the average position of a color in the scene --> used too get color dot location. 


Contains the following executables:

	clients:	- object_detection_client_node		
			- object_localisation_client_node
			- red_dot_client_node

	services:	- object_detection_service			 
			- object_localisation_service
			- red_dot_service

	publisher: 	- object_detection_publisher


Parameters can be found in the ObjectDetection.cpp.

