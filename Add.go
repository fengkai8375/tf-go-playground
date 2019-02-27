package main

import (
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "github.com/tensorflow/tensorflow/tensorflow/go/op"
    "fmt"
)

func main() {
    s := op.NewScope()

    p1 := op.Placeholder(s.SubScope("input"), tf.Int32, op.PlaceholderShape(tf.MakeShape(1, 1)))

    p2 := op.Placeholder(s.SubScope("input"), tf.Int32, op.PlaceholderShape(tf.MakeShape(1, 1)))


    p3 := op.Add(s, p1, p2)



    graph, err := s.Finalize()
    if err != nil {
        panic(err)
    }


    // Execute the graph in a session.
    sess, err := tf.NewSession(graph, nil)
    if err != nil {
        panic(err)
    }

    var matrix, column *tf.Tensor
    // A = [ [1, 2], [-1, -2] ]
    if matrix, err = tf.NewTensor([1][1]int32{{2}}); err != nil {
        panic(err.Error())
    }
    // x = [ [10], [100] ]
    if column, err = tf.NewTensor([1][1]int32{{5}}); err != nil {
        panic(err.Error())
    }
    var results []*tf.Tensor
    if results, err = sess.Run(map[tf.Output]*tf.Tensor{
        p1: matrix,
        p2: column,
    }, []tf.Output{p3}, nil); err != nil {
        panic(err.Error())
    }
    for _, result := range results {
        fmt.Println(result.Value().([][]int32))
    }
}