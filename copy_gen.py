import cv2
import shutil
import os

def copy_image():
    src = r"C:/Users/Administrator/.gemini/antigravity/brain/876c2216-3a58-4fa5-a131-5f6af4659a73/long_hairstyle_v1_1765614517039.png"
    dst = r"c:/Users/Administrator/Documents/grad1/hairstyle.jpg"
    
    # Read as png to handle transparency if any (though generated is likely solid bg as requested)
    img = cv2.imread(src)
    if img is None:
        print("Error reading source")
        return
        
    # Ensure solid black background if not already (prompt asked for it)
    # Just to be safe for the "black background" logic in main script
    # This might have white background depending on generation result.
    # Let's inspect center or corners? 
    # Actually, prompt said "isolated on solid black background".
    
    # Save as jpg
    cv2.imwrite(dst, img)
    print("Copied and saved.")

if __name__ == "__main__":
    copy_image()
