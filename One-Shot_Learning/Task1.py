
from ExampleGenerator import load_examples

def show_task_1_info(path):

    # Generate examples.
    examples = load_examples(
        path = path,
        count = 3,
        include_meta_data = True
    )

    print("***** Generated " + str(len(examples)) + " examples ***** ")
    print("")
    print("")
    for i in range(len(examples)):
        # Printing information about the first example.
        print("*** Info about example " + str(i) + " ***")
        print("image 1")
        print("   - path  = " + str(examples[i][3]))
        print("   - image = " + str(examples[i][0]))
        print("image 2")
        print("   - path  = " + str(examples[i][4]))
        print("   - image = " + str(examples[i][1]))
        print("target = " + str(examples[i][2]))
        print("")
        print("")
