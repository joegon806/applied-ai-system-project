# Model Card: Mood Machine

## 1. Limitations / Biases

The system is vastly limited by the data in its dataset: the positive and negative words, the labels, and the sample posts. Words that are obviously positive to a human may not get detected as such by the model simply because the post's positive words are not included in the data, such as "elated", "ecstatic", or even conjugated words like "happier" or "loving". The model's labels also limit the outputs it can produce, glazing over more complicated emotions like the ones in TRUE_LABELS like "sarcastic" or "passive-aggressive".

One of the system's biases is the confidence threshold. The model can still produce wrong answers if the computed uncertainty for the wrong answer is lower than the confidence threshold, so this threshold needs to be set carefully to accept both a high amount of correct posts and a low amount of incorrect posts. 

## 2. Potential Misuse

This AI model could be used for quickly evaluating the moods of human reviews/feedback of products. However, as this is a simple model that is prone to mistakes, it would be of misuse to apply this model to real-world problems as it is. To prevent this from being an issue, the model would have to be further refined with more data added to the data set and more posts tested for accuracy and confidence.
The model could also be improved by printing out what words were used to influence the label, so that the human user could verify its accuracy. This reduces the model's abstraction, however, so perhaps it could stay as a toggleable option for users who want it.

## 3. AI Collaboration
(What surprised you while testing your AI's reliability?
describe your collaboration with AI during this project. Identify one instance when the AI gave a helpful suggestion and one instance where its suggestion was flawed or incorrect.)

Something that surprised me while testing my AI's reliability was that it often yielded "uncertain" when evaluating mixed-mood posts, leaving a spot in the algorithm to be re-evaluated and improved. 
During this project, I collaborated with AI chatbot Claude Code by asking it for help designing and implementing the reliability and testing system, while I reviewed and approved or rejected its suggestions as needed. 
One instance when the AI gave a helpful suggestion was before the reliability and testing system was implemented. When I asked it for the best way to implement the system, it simultaneously addressed an issue that was present in the project, which was that the dataset had more true labels than the labels labels that the model could produce. It suggested that each of the extra labels be mapped to one of the four labels in a dictionary, so that when testing for accuracy, the tests can compare the new mapped labels to the model's predicted labels and none of the 'correct' labels would be out of scope of the model.
One instance when the AI gave a flawed suggestion was when implementing the reliability functions. The AI initially suggested for the functions to loop over a list of posts (e.g., the sample posts in dataset.py). I told the AI that instead, I wanted each function to be applicable to a single post at a time, so that they could be applied to a single post when a user enters one in main.py.
