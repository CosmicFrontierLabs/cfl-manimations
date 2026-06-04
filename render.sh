#!/bin/bash
# Unified render script for ICP animation
# Usage: ./render.sh [--final]

# Check for --final flag
FINAL_MODE=false
if [[ "$1" == "--final" ]]; then
    FINAL_MODE=true
fi

echo "=========================================="
echo "ICP Animation Render Script"
echo "=========================================="
echo ""

# Always run convergence test first
echo "Testing ICP convergence..."
python test_icp.py
echo ""

if [ "$FINAL_MODE" = true ]; then
    echo "Rendering in 4K quality (2160p, 60fps)"
    echo "WARNING: This will take several minutes!"
    echo ""
    
    # Clean any cached 4K files for fresh render
    rm -rf media/videos/icp_animation/2160p60/partial_movie_files/ 2>/dev/null
    
    # Render without preview flag for 4K
    uv run manim -qk icp_animation.py ICPAnimation
    
    VIDEO_PATH="media/videos/icp_animation/2160p60/ICPAnimation.mp4"
    
    if [ -f "$VIDEO_PATH" ]; then
        echo ""
        echo "✓ 4K render complete!"
        echo "  Video saved at: $VIDEO_PATH"
        echo ""
        echo "File info:"
        ls -lh "$VIDEO_PATH"
    else
        echo "✗ Error: 4K render may have failed"
    fi
else
    echo "Rendering in preview quality (480p, 15fps)"
    echo ""
    
    # Render with preview flag for low quality
    uv run manim -pql icp_animation.py ICPAnimation
    
    VIDEO_PATH="media/videos/icp_animation/480p15/ICPAnimation.mp4"
    
    # Also play with cvlc as backup
    if [ -f "$VIDEO_PATH" ]; then
        echo ""
        echo "✓ Preview render complete!"
        echo "  Video saved at: $VIDEO_PATH"
        echo ""
        echo "Playing with cvlc..."
        cvlc --play-and-exit "$VIDEO_PATH" 2>/dev/null || echo "cvlc not available"
    else
        echo "✗ Error: Preview render may have failed"
    fi
fi

echo ""
echo "=========================================="
echo "Usage:"
echo "  ./render.sh         # Preview quality (480p)"
echo "  ./render.sh --final # Production quality (4K)"
echo "=========================================="