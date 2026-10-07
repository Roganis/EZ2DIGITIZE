# Capturing a small object

Photogrammetry finds the same points of the object in many photos and
works out where each photo was taken from. Everything below serves that:
the same detail must be visible, sharp and lit the same way in several
photos.

## The setup

- **Light: bright and soft.** Overcast daylight, or two lamps through
  diffusers (or bounced off a white wall). Avoid hard shadows and the
  on-camera flash: shadows and highlights that move with you look like
  different surfaces.
- **A patterned mat under the object.** Newspaper, a printed pattern, a
  textured cloth. It gives COLMAP points to lock onto around the object,
  which helps most with plain objects. It gets reconstructed too; the
  object is still the part in focus.
- **For real size: the marker sheet** (3D view → Scale → Marker sheet…, or
  `ez2d markers sheet.svg`). Print it at 100 % and check the 100 mm line;
  if a black square isn't 30 mm, measure one and enter that as the marker
  size. Put the object in the middle. It doubles as a patterned mat, and
  the model comes out at its real size by itself.
- **A plain or blurred background beyond the mat** is fine. Don't let
  people or things move in the background between photos.
- **Fill the frame** with the object (and a bit of mat): two-thirds of the
  photo or more.

## Taking the photos

- **Walk all the way around**, a photo every 10 to 15° (24 to 36 photos per
  circle). Keep a similar distance.
- **Two heights at least:** one ring about level with the object (10-20°
  above the table) and one from higher up (30-45°), so the top is seen. Add
  close-ups of fine detail last.
- **Every part in at least three photos**, with neighbouring photos
  overlapping by about two-thirds.
- **Sharp photos:** good light, a fast enough shutter (1/125 s or faster
  hand-held), and enough depth of field (f/8 to f/11 on a camera; phones are
  usually fine). The photo checks flag the blurry ones.
- **Don't zoom** during a set, and keep the same camera and lens. With a
  zoom lens, tape the ring. If a set mixes cameras or zoom settings, each
  is calibrated on its own automatically, but one per set is more accurate.
- **Lock the exposure** where you can: manual mode on a camera, AE/AF lock
  on a phone (press and hold on the object). Left to itself, the camera
  brightens and darkens photos as the background changes. The photo checks
  say when it moved by more than a stop, and `ez2d photos PROJECT
  --exposure` shows by how much, and, after camera placement, whether
  those photos were placed less often.
- **Keep EXIF data** (copy the original files, don't export or resize):
  the focal length in it helps a lot.
- **iPhone:** HEIC photos are fine: a JPEG copy of each is made on import
  (the original is kept).

## Video instead of photos

Video works for a quick capture: walk slowly around the object in good
light, 30 seconds to 2 minutes. EZ2DIGITIZE keeps the sharpest frame of
each stretch (100 frames by default). Photos still give finer detail.

Videos from a GoPro (HERO5 and later, and the MAX in its single-lens
mode), and from phones and cameras that write Google's camera motion track
(CAMM), carry the camera's motion sensors. EZ2DIGITIZE reads them on
import and notes gravity's direction for every frame, so the model is
stood upright by measurement rather than by guessing how the camera was
held, which goes wrong when you shoot steeply from above or below. The
frames are also spaced by how far the camera turned rather than by time,
so slowing down or pausing on one side doesn't crowd the frames there,
and frames taken while the camera swung fast are left out.
The import says when a video has such data.

A capture app can also record the phone's motion in a file of its own,
next to the photos or the video (`<name>.motion.json`, described in
`src/ez2digitize/motion_log.py`; for a video, named after it, e.g.
`VID_0001.motion.json`). Importing the folder, or the video, picks it up,
and the photos get the same treatment as frames of a video with a motion
track. This is the format the planned Android app will write. When the
log tracked the phone's position in metres (ARCore), the model gets a
rough real size from it (within a percent or so, worse if the tracking
drifted); the marker sheet still measures it better and replaces it.

## Turntables

With the camera on a tripod and the object turning, the background stays
still and wins: COLMAP places the cameras from the background, so they all
end up in one spot (the coverage check says so after camera placement).
Use masks: in the **Masks** tab, **Make masks** separates the object from
the background in every photo (about a second per photo; `ez2d masks P`
on the command line). On the skull turntable set this is the difference
between a smeared shell and a clean model. Look through the grid: a mask
that cut off part of the object is worse than none, so uncheck it. A
plain cloth behind and under the object, lit evenly, helps both the masks
and the reconstruction. Masks made elsewhere (one PNG per photo, white for
the object) can be added with **Add masks…** or `ez2d import P photos/
--masks masks/`.

## Both sides of an object

To capture the underside too:

1. Take a full set as usual, including a low ring that shows the object's
   sides.
2. Turn the object over (upside down, or onto a side) and take a second
   set the same way, in the same light. The sides of the object must show
   in both sets: that is where the two sets are joined.
3. In the app, **Other side…** imports the second set as the turned-over
   side (`ez2d import P photos-under/ --flipped`). The **Both sides** tab
   shows which import is which side (change it there, or with `ez2d flip
   P CAPTURE`) and what is left to do.
4. Make masks for every photo (Masks tab) and look through them: no table
   or stand may be left in. Between the sets the object moved and the
   table didn't, so anything but the object pulls them apart.
5. Build. The log says whether both sides joined, and how many photos of
   each were placed. The model stands upright the way the first side was
   photographed.

## Difficult objects

- **Shiny:** reflections move with the camera and break matching. Use a
  removable matte spray (scanning spray, or dry shampoo/foot powder for a
  quick test), or cross-polarized light.
- **Transparent or very dark:** same, matte spray.
- **No texture (plain white plastic):** the patterned mat helps the camera
  placement; for the surface itself, a light pattern (masking tape bits,
  washable marker, projected dots) gives the matcher something to find.
  If too few photos are placed, try **Advanced → Features: ALIKED +
  LightGlue**: learned features find more on weak texture (slower).
- **Thin parts** (wires, hair, leaves): they will be thick or missing at
  any quality; High helps a little.

## Rooms and outdoor scenes

The app is made for small objects, but a room, a building front or a
garden works too. Set **Subject** to **Room or outdoor scene** (`ez2d new
--scene`, or `ez2d run --subject scene`). Masks are then not used, the
camera rings and their advice are left out (they assume photos all round
an object), and plain walls and floors are kept when meshing.

- **Walk, don't spin.** Standing in one spot and turning gives views with
  no depth between them. Take a step sideways between photos, so each
  overlaps the last by about two thirds.
- **In a room,** walk along the walls looking across and inwards, then
  through the middle; add photos looking up at the ceiling and down at the
  floor near the walls. Plain walls with nothing on them are hard: posters,
  furniture or a few sticky notes help.
- **Outdoors,** keep the light the same (overcast is best) and leave out
  moving things (people, cars, trees in wind) where you can.
- **Many photos are fine.** Up to 200 photos, every pair is compared.
  Beyond that, phone photos with a GPS position are compared with their
  neighbours, and others with the photos that look most alike. That needs
  COLMAP's vocabulary tree, which is downloaded once. Without it, each
  photo is compared with those taken just before and after it, so take
  them in order then. Videos are compared frame by frame, with the tree
  finding where a walk comes back to its start; a video whose motion track
  records where the camera was (CAMM, from ARCore-style tracking apps)
  compares each frame with those seeing the same side instead.
- **Size and memory.** Large scenes need far more memory in the dense
  step. Start with Fast, and use the crop box to keep only the part you
  want.

## After a run

The log notes what the reconstruction saw: how many photos were placed,
gaps around the object, all photos from one height, photos placed far from
the rest. They are hints for the next capture.
