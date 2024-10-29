class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next


def twist_linked_list(head):
    if not head or not head.next:
        return head

    # 将链表节点存入数组
    nodes = []
    current = head
    while current:
        nodes.append(current)
        current = current.next

    # 创建新链表
    new_head = ListNode(0)
    new_current = new_head
    left, right = 0, len(nodes) - 1

    # 交替插入节点
    while left <= right:
        if left == right:
            new_current.next = nodes[left]
            new_current = new_current.next
        else:
            new_current.next = nodes[right]
            new_current = new_current.next
            new_current.next = nodes[left]
            new_current = new_current.next
        left += 1
        right -= 1

    new_current.next = None
    return new_head.next


# 测试代码
def test_twist_linked_list(values):
    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next

    result = twist_linked_list(head)
    output = []
    while result:
        output.append(result.value)
        result = result.next
    print("{" + ", ".join(map(str, output)) + "}")


# 测试示例
test_twist_linked_list([1, 2, 3, 4, 5])  # 输出: {5, 3, 1, 2, 4}
