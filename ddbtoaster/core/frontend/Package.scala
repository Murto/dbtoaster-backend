package ddbt

import ddbt.ast.M3._

package object frontend {
  type DependencyGraph = Map[Statement, Set[Statement]]
}
