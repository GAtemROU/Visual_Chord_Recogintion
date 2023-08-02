# Visual_Chord_Recogintion
Project was done by two students Tymur Mykhalievskyi and Alexander Kuehn from University of Saarland.

The main goal of the project was to recognize 14 different chords: 7 major and 7 minor.

We've applied transfer learning in our problem. The concept is not new, it is simply to take pretrained network and apply it on our problem. 
What was different in our case is that training on the whole data lead to 20% less accuracy comparing to making an intermidiate step in the middle. This step is to firstly train network and less diverse data and then switch to the whole problem. This lead to 20% improvement on validation data.

Our final model gets 90%+ accuracy on the simple test created from the images of same person we had during training and 60% on the hard test which was created from the completely unseen data.

Check the project slides for more info.

Final report is coming soon.
