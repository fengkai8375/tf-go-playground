package main

import (
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "github.com/tensorflow/tensorflow/tensorflow/go/op"
    "fmt"
)

func main() {
    s := op.NewScope()

    p1 := op.Placeholder(s.SubScope("input"), tf.Int32, op.PlaceholderShape(tf.MakeShape(2, 2)))

    p2 := op.Placeholder(s.SubScope("input"), tf.Int32, op.PlaceholderShape(tf.MakeShape(2, 2)))


    p3 := op.MatMul(s, p1, p2)



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
    // p1 = [ [1, 2], [-1, -2] ]
    if matrix, err = tf.NewTensor([2][2]int32{{1, 2}, {-1, -3}}); err != nil {
        panic(err.Error())
    }
    // p2 = [ [10], [100] ]
    if column, err = tf.NewTensor([2][2]int32{{10,2}, {100,1}}); err != nil {
        panic(err.Error())
    }
    var results []*tf.Tensor
    results, err = sess.Run(map[tf.Output]*tf.Tensor{
        p1: matrix,
        p2: column,
    }, []tf.Output{p3}, nil)
    if err != nil {
        panic(err.Error())
    }

    for _, result := range results {
        r1 := result.Value().([][]int32)
        fmt.Println(r1)
    }
}