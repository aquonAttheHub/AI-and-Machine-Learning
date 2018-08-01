{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "Intro to Tensorflow\n",
    "---\n",
    "\n",
    "Import tensorflow and other needed libraries here:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "b'Hello, TensorFlow!'\n"
     ]
    }
   ],
   "source": [
    "# TODO: Add import tensorflow.\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "hello = tf.constant('Hello, TensorFlow!')\n",
    "sess = tf.Session()\n",
    "print(sess.run(hello))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Clear the Graph\n",
    "---\n",
    "Clear the default graph to reset everything back to default"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Clear the tensorflow graph\n",
    "tf.reset_default_graph()\n",
    "test_constant = tf.constant(10.0, dtype=tf.float32)\n",
    "add_one_operation = test_constant + 1\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Set up Placeholders\n",
    "---\n",
    "Placeholders are places you can feed data into your model. They indicate a value that you'll feed in later when the network is run."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Create placeholders\n",
    "tf.reset_default_graph()\n",
    " \n",
    "input_data = tf.placeholder(dtype=tf.float32, shape=[None, 2])\n",
    " \n",
    "double_operation = input_data * 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Variables\n",
    "---\n",
    "Variables keep their value between runs. These are the building blocks of machine learning, and represent the parameters that will be tuned as your model trains"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Create variables\n",
    "tf.reset_default_graph()\n",
    "input_data = tf.placeholder(dtype=tf.float32, shape=None)\n",
    "output_data = tf.placeholder(dtype=tf.float32, shape=None)\n",
    "slope=tf.Variable(5, dtype=tf.float32)\n",
    "intercept = tf.Variable(2, dtype=tf.float32)\n",
    "model_operation = slope * input_data + intercept\n",
    "\n",
    "error = model_operation - output_data\n",
    "squared_error = tf.square(error)\n",
    "loss = tf.reduce_mean(squared_error)\n",
    "\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)\n",
    "train = optimizer.minimize(loss)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Running a Session\n",
    "---\n",
    "Once your graph is built, you can start a session. Putting the session in a `with` statement allows the session to automatically close once the statement finishes:\n",
    "```\n",
    "with tf.Session() as sess:\n",
    "    # Run the session in here\n",
    "    ...\n",
    "# Session closes when you get here\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n[5.0, 2.0]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[5.0, 2.0]\n0.0\n"
     ]
    },
    {
     "data": {
      "image/png": "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\nal4AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f833aa1d198>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# TODO: Run a session\n",
    "\n",
    "init = tf.global_variables_initializer()\n",
    "#this will reset all variables back to their initial values.\n",
    "\n",
    "x_values = [0, 1, 2, 3,  4]\n",
    "y_values = [2, 7, 12, 17, 22]\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    for i in range(2000):\n",
    "        sess.run(train, feed_dict={input_data:x_values, output_data:y_values})\n",
    "        if i % 100 == 0:\n",
    "            print(sess.run([slope, intercept]))\n",
    "            plt.plot(x_values, sess.run(model_operation, feed_dict={input_data:x_values}))\n",
    "            \n",
    "    print(sess.run(loss, feed_dict={input_data:x_values, output_data:y_values}))\n",
    "    plt.plot(x_values, y_values, 'ro', 'Training Data')\n",
    "    plt.plot(x_values, sess.run(model_operation, feed_dict={input_data:x_values}))\n",
    "     \n",
    "    plt.show()                  \n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Neurons and Neural Networks\n",
    "---\n",
    "Tensorflow provides functions to easily create layers of neurons for a neural network."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
