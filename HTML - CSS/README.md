# Note on HTML and CSS in Gradio

When using **Gradio** to build interactive web interfaces in Python, it's important to understand how the **CSS** and **HTML** work.

- The **CSS in Gradio is external**, which means you **do NOT need to include CSS stylesheets manually** in your project. Gradio automatically applies its styling for you.

- However, to **activate the HTML and render the Gradio interface correctly**, you **_must_** run the Gradio app in an environment that supports web serving, such as:

  - **Locally on your machine** with `app.launch()`
  - Or **_must_** run the Gradio notebook code in **Google Colab** or similar notebook environments that support web output.

> **_Important:_**  
> You **must** run the Gradio app (e.g., `app.launch()`) to see and interact with the HTML interface.

---

### Run Gradio on Google Colab

Here is a useful [Google Colab example notebook for Gradio](https://colab.research.google.com/github/gradio-app/gradio/blob/main/demo/Quickstart.ipynb) that demonstrates how to create and launch Gradio interfaces easily.

Feel free to open and run it directly in Colab!

---

If you have any questions or need further help, don’t hesitate to ask.
