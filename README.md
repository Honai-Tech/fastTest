## Inspirations
With the pandemic situation currently, the need of getting medical attention from medical professionals increases, while the access becomes limited. Many times, avoiding certain foods can get rid of light symptoms without going to a clinic. We built an interface that allows the users to get an idea of foods that need to be avoided based on the symptoms they have.

## What it does
fasTest gives an quick recommendation to the users to avoid certain foods that potentially can cause symptoms experienced by the users. 

## How I built it
We utilized google colaboratory to develop the source code, which mainly based on python with pandas and numpy libraries. We uses Random Forest classifer to develop a machine learning model for food prediction that is based on input symptoms. Finally, the model is deployed on the web using Gradio.app. 

## Challenges I ran into
Some challenges we had are including cleaning up the data, and getting a higher prediction accuracy of the machine learning model. 

## Accomplishments that I'm proud of
We are able to deploy the model, with input of three symptoms and output of food to be avoided.

## What I learned
We learned to use machine learning techniques. 

## What's next for fasTest
Getting a higher accuracy model and enable user input without choices. 

## Try it out
https://54654.gradio.app/