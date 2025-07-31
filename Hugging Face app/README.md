## Hugging Face Spaces: Deploying the Model with Gradio

Hugging Face Spaces provides a user-friendly cloud platform for hosting and deploying machine learning models and interactive demos. It supports popular frontend tools like Gradio and Streamlit, enabling developers and researchers to share their work seamlessly with the public or collaborators.

In this project, we used **Gradio** within Hugging Face Spaces to create an interactive interface that allows users to input their data, run the model behind the scenes, and receive the output in real time. The app is connected to a GitHub repository, which ensures that every code update is automatically reflected in the live demo.

One of the main benefits of using Spaces is the speed and simplicity of deployment. Unlike traditional infrastructure that requires setting up servers, Docker containers, or cloud environments manually, Hugging Face Spaces allows you to deploy a fully functional app within minutes — directly from your browser or Git command line.

However, it's important to note that Spaces have some limitations, such as limited compute resources and runtime constraints on free tiers. For heavier models or long-running tasks, Hugging Face also offers **paid tiers** or integration with **Inference Endpoints** for scalable, production-grade deployment

Link - https://huggingface.co/spaces/sahar-yaccov/Microsoft-Attack
