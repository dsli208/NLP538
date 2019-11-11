<p>This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)</p>
<h1>Credits:</h1>
<p>Most of this code has been directly taken from the authors of:</p>
<blockquote>
<p>From A Fast and Accurate Dependency Parser using Neural Networks (2014, Danqi and Manning)</p>
</blockquote>
<p>We have adapted it to work with tensorflow2.0, and changed it to similar format as assignment 2. Also thanks to previous TA, Heeyoung Kwon who set up the original assignment.</p>
<h1>Overview</h1>
<p>You will implement a neural Dependency Parsing model by writing code for the following:</p>
<p>From Incrementality in Deterministic Dependency Parsing (2004, Nivre)</p>
<ul>
<li>the arc-standard algorithm</li>
</ul>
<p>From A Fast and Accurate Dependency Parser using Neural Networks (2014, Danqi and Manning)</p>
<ul>
<li>feature extraction</li>
<li>the neural network architecture including activation function</li>
<li>loss function</li>
</ul>
<h1>Installation</h1>
<p>The environment is same as assignment 2. But we would <em>strongly</em> encourage you to make a new environment for assignment 3.</p>
<p>This assignment is implemented in python 3.6 and tensorflow 2.0. Follow these steps to setup your environment:</p>
<ol>
<li><a href="http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html" title="Download and install Conda">Download and install Conda</a></li>
<li>Create a Conda environment with Python 3.6</li>
</ol>
<pre><code>conda create -n nlp-hw3 python=3.6
</code></pre>
<ol start="3">
<li>Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.</li>
</ol>
<pre><code>conda activate nlp-hw3
</code></pre>
<ol start="4">
<li>Install the requirements:</li>
</ol>
<pre><code>pip install -r requirements.txt
</code></pre>
<ol start="5">
<li>Download glove wordvectors:</li>
</ol>
<pre><code>./download_glove.sh
</code></pre>
<p><strong>NOTE:</strong> We will be using this environment to check your code, so please don't work in your default or any other python environment.</p>
<h1>Data</h1>
<p>You have training, development and test set for dependency parsing in conll format. The <code>train.conll</code> and <code>dev.conll</code> are labeled whereas <code>test.conll</code> is unlabeled</p>
<p>For quick code development/debugging, this time we have explicitly provided small fixture dataset. You can use this as training and development dataset while working on the code.</p>
<h1>Code Overview</h1>
<p>This repository largely follows the same interface as assignment 2.</p>
<h2>Train, Predict, Evaluate</h2>
<p>You have three scripts in the repository <code>train.py</code>, <code>predict.py</code> and <code>evaluate.py</code> for training, predicting and evaluating a Dependency Parsing Model. You can supply <code>-h</code> flag to each of these to figure out how to use these scripts.</p>
<p>Here we show how to use the commands with the defaults.</p>
<h4>Train a model</h4>
<pre><code>python train.py data/train.conll data/dev.conll

# stores the model by default at : serialization_dirs/default
</code></pre>
<h4>Predict with model</h4>
<pre><code>python predict.py serialization_dirs/default \
                  data/dev.conll \
                  --predictions-file dev_predictions.conll
</code></pre>
<h4>Evaluate model predictions</h4>
<pre><code>python evaluate.py serialization_dirs/default \
                   data/dev.conll \
                   dev_predictions.conll
</code></pre>
<p><strong>NOTE:</strong> These scripts will not work until you fill-up the placeholders (TODOs) left out as part of the assignment.</p>
<h2>Dependency Parsing</h2>
<ul>
<li>
<p><code>lib.model:</code> Defines the main model class of the neural dependency parser.</p>
</li>
<li>
<p><code>lib.data.py</code>: Code dealing with reading, writing connl dataset, generating batches, extracting features and loading pretrained embedding file.</p>
</li>
<li>
<p><code>lib.dependency_tree.py</code>: The dependency tree class file.</p>
</li>
<li>
<p><code>lib.parsing_system.py</code>: This file contains the class for a transition-based parsing framework for dependency parsing.</p>
</li>
<li>
<p><code>lib.configuration.py</code>: The configuration class file. Configuration reflects a state of the parser.</p>
</li>
<li>
<p><code>lib.util.py</code>: This file contain function to load pretrained Dependency Parser.</p>
</li>
<li>
<p><code>constants.py</code>: Sets project-wide constants for the project.</p>
</li>
</ul>
<h1>Expectations</h1>
<h2>What to write in code:</h2>
<p>Like assignment 2 you have <code>TODO(Students) Start</code> and <code>TODO(Students) End</code> annotations. You are expected to write your code between those comment/annotations.</p>
<ol>
<li>Implement the arc-standard algorithm in <code>parsing_system.py</code>: <code>apply</code> method</li>
<li>Implement feature extraction in <code>data.py</code>: <code>get_configuration_features</code> method</li>
<li>Implement neural network architecture in <code>model.py</code> in <code>DependencyParser</code> class: <code>__init__</code> and <code>call</code> method.</li>
<li>Implement loss function for neural network in <code>model.py</code> in <code>DependencyParser</code> class: <code>compute_loss</code> method.</li>
</ol>
<h2>What experiments to try</h2>
<p>You should try experiments to figure out the effects of following on learning:</p>
<ol>
<li>activations (cubic vs tanh vs sigmoid)</li>
<li>pretrained embeddings</li>
<li>tunability of embeddings</li>
</ol>
<p>and write your findings in the report.</p>
<p>The file <code>experiments.sh</code> enlists the commands you will need to train and save these models. In all you will need ~5 training runs, each taking about 30 minutes on cpu. See <code>colab_notes.md</code> to run experiments on gpu.</p>
<p>As shown in the <code>experiments.sh</code>, you can use <code>--experiment-name</code> argument in the <code>train.py</code> to store the models at different locations in <code>serialization_dirs</code>. You can also use <code>--cache-processed-data</code> and <code>--use-cached-data</code> flags in <code>train.py</code> to not generate the training features everytime. Please look at training script for details. Lastly, after training your dev results will be stored in serialization directory of the experiment with name <code>metric.txt</code>.</p>
<p><strong>NOTE</strong>: You will be reporting the scores on development set and submitting to use the test prediction of the configuration that you found the best. The labels of test dataset are hidden from you.</p>
<h2>What to turn in?</h2>
<p>A single zip file containing the following files:</p>
<ol>
<li>parsing_system.py</li>
<li>data.py</li>
<li>model.py</li>
<li>test_predictions.conll</li>
<li>gdrive_link.txt</li>
<li>report.pdf</li>
</ol>
<p><code>gdrive_link.txt</code> should have a link to the <code>serialization_dirs.zip</code> of your trained models.</p>
<p>We will release the exact zip format on piazza in a couple of days.</p>
<h3>Good Luck!</h3>
