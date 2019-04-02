echo "You are beautiful, give us hundred"

# make run path="blah blah"
run:
	python proposedAlgorithm.py $(path)
	python detect_shapes.py -i ye.png
	python text_recognition.py --east frozen_east_text_detection.pb --image ye.png
	python text_recognition.py --east frozen_east_text_detection.pb --image $(path)

git:
	git add -A
	git commit -m "[QUICK COMMIT]"
	git push