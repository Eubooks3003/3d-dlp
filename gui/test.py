def test_function():
    import matplotlib.pyplot as plt
    def transform_keypoints(original_keypoints, modified_keypoints):
        # Convert the keypoints to numpy arrays
        p0, p1, p2, p3 = np.array(original_keypoints)
        p0_new, p3_new = np.array(modified_keypoints)

        # Calculate translation vectors
        translation_original = p0
        translation_new = p0_new

        # Calculate the translation matrix
        translation_matrix = np.eye(3)
        translation_matrix[:2, 2] = translation_new - translation_original

        # Calculate rotation and scaling parameters
        angle_original = np.arctan2(p3[1] - p0[1], p3[0] - p0[0])
        angle_new = np.arctan2(p3_new[1] - p0_new[1], p3_new[0] - p0_new[0])
        scale_original = np.linalg.norm(p3 - p0)
        scale_new = np.linalg.norm(p3_new - p0_new)

        # Calculate the rotation matrix
        rotation_matrix = np.eye(3)
        rotation_matrix[:2, :2] = [[np.cos(angle_new - angle_original), -np.sin(angle_new - angle_original)],
                                   [np.sin(angle_new - angle_original), np.cos(angle_new - angle_original)]]

        # Calculate the scaling matrix
        scaling_matrix = np.eye(3)
        scaling_matrix[0, 0] = scale_new / scale_original
        scaling_matrix[1, 1] = scale_new / scale_original

        # Combine the matrices to obtain the final transformation matrix
        transformation_matrix = np.matmul(scaling_matrix, np.matmul(rotation_matrix, translation_matrix))

        # Apply the transformation to p1 and p2
        p1_transformed = np.matmul(transformation_matrix, np.append(p1, 1))[:2]
        p2_transformed = np.matmul(transformation_matrix, np.append(p2, 1))[:2]

        return p1_transformed.tolist(), p2_transformed.tolist()

    original_keypoints = [(10, 10), (25, 25), (32, 20), (40, 40)]
    modified_keypoints = [(5, 5), (40, 40)]

    p1_transformed, p2_transformed = transform_keypoints(original_keypoints, modified_keypoints)

    # Unpack the original keypoints
    p0, p1, p2, p3 = original_keypoints

    # Create a figure and axis objects
    fig, ax = plt.subplots()

    # Plot the original keypoints
    ax.plot([p0[0], p3[0]], [p0[1], p3[1]], 'r-', label='Original Trajectory')
    ax.plot(p0[0], p0[1], 'ro', label='p0')
    ax.plot(p1[0], p1[1], 'ro', label='p1')
    ax.plot(p2[0], p2[1], 'ro', label='p2')
    ax.plot(p3[0], p3[1], 'ro', label='p3')

    p0_new, p3_new = modified_keypoints
    # Plot the modified keypoints
    ax.plot([p0_new[0], p3_new[0]], [p0_new[1], p3_new[1]], 'g-', label='Modified Trajectory')
    ax.plot(p0_new[0], p0_new[1], 'go', label='p0_new')
    ax.plot(p3_new[0], p3_new[1], 'go', label='p3_new')

    # Plot the transformed keypoints
    ax.plot(p1_transformed[0], p1_transformed[1], 'bo', label='p1_transformed')
    ax.plot(p2_transformed[0], p2_transformed[1], 'bo', label='p2_transformed')

    # Set axis limits and labels
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()
